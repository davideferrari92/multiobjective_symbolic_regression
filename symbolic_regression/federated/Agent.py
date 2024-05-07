import logging
import pickle
from datetime import datetime
from multiprocessing.connection import Client
from typing import Dict, List
from symbolic_regression.SymbolicRegressor import SymbolicRegressor
from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase

from symbolic_regression.federated.Communication import \
    FederatedDataCommunication
from symbolic_regression.federated.strategies.BaseStrategy import BaseStrategy


class FederatedAgent:

    def __init__(self, name: str, address: str, port: int, orchestrator_address: str, orchestrator_port: int, save_path: str = None, callbacks: List[MOSRCallbackBase] = list()) -> None:
        """ This class implement a Federated Agent

        Args:
            - name: str
                Name of the agent
            - address: str
                Address of the agent
            - port: int
                Port of the agent
            - orchestrator_address: str
                Address of the orchestrator
            - orchestrator_port: int
                Port of the orchestrator
            - save_path: str
                Path to save the results

        Attributes:
            - name: str
                Name of the agent
            - address: str
                Address of the agent
            - port: int
                Port of the agent
            - orchestrator_address: str
                Address of the orchestrator
            - orchestrator_port: int
                Port of the orchestrator
            - mode: str
                Mode of the agent (client or server)
            - is_registered: bool
                True if the agent is registered to the orchestrator
            - _status: str
                Status of the agent. Will trigger a sync with the orchestrator if changed
            - _logs: list
                List of logs
            - clients: dict
                Dictionary of clients
            - servers: dict
                Dictionary of servers
            - training_configuration: dict
                Dictionary of training configuration
            - _federated_aggregation_strategy: BaseStrategy
                Strategy used for federated aggregation
            - save_path: str
                Path to save the results

        Returns:
            - None
        """
        self.name: str = name
        self.address: str = address
        self.port: int = port
        self.orchestrator_address: str = orchestrator_address
        self.orchestrator_port: int = orchestrator_port
        self.mode: str = None

        self.is_registered: bool = False
        self._status: str = 'idle'
        self._logs: List = list()

        self.clients: Dict[str, Dict] = dict()
        self.servers: Dict[str, Dict] = dict()

        self.training_configuration: Dict = dict()
        self._federated_aggregation_strategy: BaseStrategy = None

        self.save_path = save_path

        self.callbacks = callbacks

    #############################################
    # Properties
    #############################################

    @property
    def federated_aggregation_strategy(self):
        return self._federated_aggregation_strategy

    @federated_aggregation_strategy.getter
    def federated_aggregation_strategy(self) -> BaseStrategy:
        return self._federated_aggregation_strategy

    @federated_aggregation_strategy.setter
    def federated_aggregation_strategy(self, new_strategy: BaseStrategy):
        if not self.mode:
            raise AttributeError('Mode not set for this agent')
        logging.info(f'Setting federated aggregation strategy: {new_strategy}')
        self._federated_aggregation_strategy: BaseStrategy = new_strategy
        logging.info(
            f'Setting federated aggregation strategy mode: {self.mode}')
        self._federated_aggregation_strategy.mode = self.mode
        self._federated_aggregation_strategy.name = self.name

        if isinstance(self._federated_aggregation_strategy.regressor, SymbolicRegressor):
            self._federated_aggregation_strategy.regressor.callbacks = self.callbacks

    @property
    def status(self):
        return self._status

    @status.getter
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, new_status: str):
        logging.info(f'Setting status: {new_status}')
        self._status: str = new_status
        self.sync_status()

    #############################################
    # Operational
    #############################################

    def log_activity(self, agent_name: str, activity: str, details: str = '') -> None:
        """ Log an activity

        Args:
            - agent_name: str
                Name of the agent
            - activity: str
                Activity
            - details: dict
                Details of the activity

        Returns:
            - None
        """

        if self.is_client(agent_name):
            mode = 'client'
        elif self.is_server(agent_name):
            mode = 'server'
        elif self.mode == 'orchestrator':
            mode = 'orchestrator'
        else:
            logging.warning(f'Logging event of unknown agent: {agent_name}')
            mode = 'unknown'

        created = datetime.now()

        self._logs.append({
            'agent_name': agent_name,
            'agent_address': self.clients[agent_name]['address'] if mode == 'client' else self.servers[agent_name]['address'],
            'agent_port': self.clients[agent_name]['port'] if mode == 'client' else self.servers[agent_name]['port'],
            'mode': mode,
            'activity': activity,
            'details': details,
            'created': created,
        })

    def load(self, path: str = None) -> None:
        """ Load the state of the agent

        Args:
            - path: str
                The path where to load the state

        Returns:
            - None
        """

        with open(path if path else self.save_path, 'rb') as f:
            loaded_orchestrator = pickle.load(f)

        return loaded_orchestrator

    def save(self, path: str = None) -> None:
        """ Save the state of the agent

        Args:
            - path: str
                The path where to save the state

        Returns:
            - None
        """

        with open(path if path else self.save_path, 'wb') as f:
            pickle.dump(self, f)

    #############################################
    # Communication
    #############################################

    def _what_mode(self, agent_name: str) -> str:
        """ Return the mode of the agent

        Args:
            - agent_name: str
                Name of the agent

        Returns:
            - str
                Mode of the agent
        """
        if agent_name in self.clients:
            return 'client'
        elif agent_name in self.servers:
            return 'server'
        else:
            logging.warning(
                f'Agent {agent_name} is not registered in the orchestrator')
            return None

    def is_client(self, agent_name: str) -> bool:
        """ Return True if the agent is a client

        Args:
            - agent_name: str
                Name of the agent

        Returns:
            - bool
                True if the agent is a client
        """
        return self._what_mode(agent_name) == 'client'

    def is_server(self, agent_name: str) -> bool:
        """ Return True if the agent is a server

        Args:
            - agent_name: str
                Name of the agent

        Returns:
            - bool
                True if the agent is a server
        """
        return self._what_mode(agent_name) == 'server'

    def is_orchestrator(self, agent_name: str) -> bool:
        """ Return True if the agent is the orchestrator

        Args:
            - agent_name: str
                Name of the agent

        Returns:
            - bool
                True if the agent is the orchestrator
        """
        return agent_name == 'orchestrator'

    def broadcast(self, comm_type: str, payload: object = None) -> None:
        """ Broadcast a message to all agents

        Args:
            - comm_type: str
                Type of the message
            - payload: object
                Payload of the message

        Returns:
            - None
        """
        self.send_to_all_clients(comm_type=comm_type, payload=payload)
        self.send_to_all_servers(comm_type=comm_type, payload=payload)

    def send_to_orchestrator(self, comm_type: str, payload: object = None, attempts: int = 5) -> None:
        """ Send a message to the orchestrator

        Args:
            - comm_type: str
                Type of the message
            - payload: object
                Payload of the message

        Returns:
            - None
        """
        if not self.orchestrator_address or not self.orchestrator_port:
            raise AttributeError('Orchestrator not set')

        for att in range(attempts):
            logging.debug(
                f'Sending {comm_type} to Orchestrator ({att+1}/{attempts})')
            try:
                conn = Client(
                    (self.orchestrator_address, self.orchestrator_port))
                conn.send(
                    FederatedDataCommunication(
                        sender_name=self.name,
                        sender_address=self.address,
                        sender_port=self.port,
                        comm_type=comm_type,
                        payload=payload
                    )
                )
                conn.close()
                return

            except ConnectionResetError:

                self.log_activity(
                    agent_name=self.name,
                    activity='ConnectionResetError',
                    details='ConnectionResetError while sending message'
                )
                logging.warning(
                    f'ConnectionResetError: Orchestrator is not reachable')

            except ConnectionRefusedError:
                self.log_activity(
                    agent_name=self.name,
                    activity='ConnectionRefusedError',
                    details='ConnectionRefusedError while sending message'
                )
                logging.warning(
                    f'ConnectionRefusedError: Orchestrator is not reachable')

            except TimeoutError:
                self.log_activity(
                    agent_name=self.name,
                    activity='TimeoutError',
                    details='TimeoutError while sending message'
                )
                logging.warning(
                    f'TimeoutError: Orchestrator is not reachable')

        if comm_type == 'SyncStatus':
            logging.debug(
                f'SyncStatus {payload} sent to Orchestrator ({self.orchestrator_address}:{self.orchestrator_port})')
        else:
            logging.debug(
                f'Message {comm_type} sent to Orchestrator ({self.orchestrator_address}:{self.orchestrator_port})')

    def send_to_all_servers(self, comm_type: str, payload: object = None) -> None:
        """ Send a message to all servers

        Args:
            - comm_type: str
                Type of the message
            - payload: object
                Payload of the message  

        Returns:
            - None         
        """
        logging.info(
            f'Sending message {comm_type} to all servers: {", ".join(list(self.servers.keys()))}')

        for fed_server in self.servers:
            self._send_to_agent(
                agent_name=fed_server,
                comm_type=comm_type,
                payload=payload
            )

    def send_to_all_clients(self, comm_type: str, payload: object = None) -> None:
        """ Send a message to all clients

        Args:
            - comm_type: str
                Type of the message
            - payload: object
                Payload of the message

        Returns:
            - None
        """
        logging.info(
            f'Sending message {comm_type} to all clients: {", ".join(list(self.clients.keys()))}')

        for fed_client in self.clients:
            self._send_to_agent(
                agent_name=fed_client,
                comm_type=comm_type,
                payload=payload
            )

    def _send_to_agent(self, agent_name: str, comm_type: str, payload: object = None, attempts: int = 5) -> None:
        """ Send a message to a specific agent

        Args:
            - agent_name: str
                Name of the agent to send the message
            - comm_type: str
                Type of the message
            - payload: object
                Payload of the message

        Returns:
            - None
        """
        if not self.clients.get(agent_name) and not self.servers.get(agent_name):
            logging.warning(f'Agent {agent_name} not registered')
            return

        for att in range(attempts):
            logging.debug(
                f'Sending message {comm_type} to {agent_name} ({att+1}/{attempts})') 
            try:
                if self.is_client(agent_name):
                    agent_type = 'client'
                    conn = Client(
                        (self.clients[agent_name]['address'], self.clients[agent_name]['port']))
                elif self.is_server(agent_name):
                    agent_type = 'server'
                    conn = Client(
                        (self.servers[agent_name]['address'], self.servers[agent_name]['port']))
                elif self.is_orchestrator(agent_name):
                    agent_type = 'orchestrator'
                    conn = Client(
                        (self.orchestrator_address, self.orchestrator_port))

                conn.send(
                    FederatedDataCommunication(
                        sender_name=self.name,
                        sender_address=self.address,
                        sender_port=self.port,
                        comm_type=comm_type,
                        payload=payload
                    )
                )

                conn.close()

                logging.debug(
                    f'Message {comm_type} sent to {agent_name} ({agent_type})')
                return

            except ConnectionResetError:

                self.log_activity(
                    agent_name=agent_name,
                    activity='ConnectionResetError',
                    details='ConnectionResetError while sending message'
                )
                logging.warning(
                    f'ConnectionResetError: Agent {agent_name} is not reachable')

            except ConnectionRefusedError:
                self.log_activity(
                    agent_name=agent_name,
                    activity='ConnectionRefusedError',
                    details='ConnectionRefusedError while sending message'
                )
                logging.warning(
                    f'ConnectionRefusedError: Agent {agent_name} is not reachable')

            except TimeoutError:
                self.log_activity(
                    agent_name=agent_name,
                    activity='TimeoutError',
                    details='TimeoutError while sending message'
                )
                logging.warning(
                    f'TimeoutError: Agent {agent_name} is not reachable')

    def sync_status(self):
        """ Send the status of the client to the orchestrator

        Returns:
            - None
        """
        self.send_to_orchestrator(comm_type='SyncStatus', payload=self.status)
