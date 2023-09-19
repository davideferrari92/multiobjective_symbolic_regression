import logging
from datetime import datetime
from multiprocessing.connection import Listener
from typing import Dict, List

from symbolic_regression.federated.Agent import FederatedAgent
from symbolic_regression.federated.Communication import \
    FederatedDataCommunication
from symbolic_regression.federated.strategies.BaseStrategy import BaseStrategy


class Orchestrator(FederatedAgent):

    def __init__(self, name: str, address: str, port: int, training_configuration: Dict, save_path: str = None) -> None:
        """
        This class implement a Federated Symbolic Regression Orchestrator

        An orchestrator is an agent that implements the orchestration side of the
        Federated Learning algorithm. It contains the training configuration and
        the aggregation strategy. It is responsible for the orchestration of the
        training and aggregation of the models.
        The orchestrator is developed to support the following communications from the clients:

            - RegisterClient
                To register a client to the orchestrator
            - RegisterServer
                To register a server to the orchestrator
            - SyncStatus
                To sync clients and servers and monitor theis status
            - TerminationValidation
                When clients terminated the training and validated all the performance metrics
                of the strategies of all other clients on their dataset
            - ToOrchestratorAggregationStrategy
                WHen the server or a client sends the aggregation strategy to the orchestrator
                Based on who sends the message, the orchestrator will update the list of
                federated_aggregation_strategies (for the client) or the federated_aggregation_strategy
                (for the server) and, if necessary, trigger the new training of the clients.
            - UnregisterClient
                When a client requires to unregister from the orchestrator
            - UnregisterServer
                When a server requires to unregister from the orchestrator

        Args:
            - name: str
                Name of the orchestrator
            - address: str
                Address of the orchestrator
            - port: int 
                Port of the orchestrator
            - training_configuration: Dict
                Training configuration
            - save_path: str
                Path to save the orchestrator
        """

        super().__init__(name=name, address=address, port=port,
                         orchestrator_address=address, orchestrator_port=port, save_path=save_path)

        # Communication
        self.mode: str = 'orchestrator'

        self.training_configuration = training_configuration

        # Operational
        self.is_registered: bool = True  # Orchestrator is always registered
        self.federated_aggregation_strategies: Dict[str, BaseStrategy] = {}
        self._federated_aggregation_strategies_history: List[Dict[str, BaseStrategy]] = [
        ]
        self._federated_aggregation_strategy_history: List[BaseStrategy] = []

        self.validated_strategies: Dict[str, BaseStrategy] = {}

    def register(self) -> None:
        """ Register the orchestrator to the orchestrator

        Args:
            - None

        Returns:
            - None
        """
        self.is_registered = True

        try:
            self.run_orchestrator()
        except KeyboardInterrupt:
            logging.info('Orchestrator stopped by user')

    def run_orchestrator(self):
        listener = Listener((self.address, self.port))

        logging.info(
            f'Orchestrator is listening on {self.address}:{self.port}')

        while True:
            self.save()
            """ Wait for a message from a client or a server

            For a better execution flow, we recommend to invoke any sender method
            as last instruction of each branch of the if statement.
            This will allow the listener to immediately be ready to receive
            the next message.
            """
            try:
                conn = listener.accept()
                msg: FederatedDataCommunication = conn.recv()
            except EOFError:
                logging.error('EOFError')
                continue

            if msg.comm_type == 'RegisterClient':
                """
                From 1 Client
                To 1 Client
                """
                self._register_client(msg=msg)

            elif msg.comm_type == 'RegisterServer':
                """
                From 1 Server
                To 1 Server
                """
                self._register_server(msg=msg)

            elif msg.comm_type == 'SyncStatus':
                """
                From 1 Client or Server
                To None
                """
                self._receive_sync_status(msg=msg)

            elif msg.comm_type == 'TerminationValidation':

                logging.debug(
                    f'Receiving termination validation from {msg.sender_name}')
                
                # self.federated_aggregation_strategy = msg.payload['server']
                # self._federated_aggregation_strategy_history.append(
                #     self.federated_aggregation_strategy)
                self.federated_aggregation_strategies[msg.sender_name] = msg.payload#['clients']
                self._federated_aggregation_strategies_history.append(
                    self.federated_aggregation_strategies)

                self.save()

            elif msg.comm_type == 'ToOrchestratorAggregationStrategy':
                """
                From 1 Client
                To None
                """
                if self.is_client(msg.sender_name):
                    logging.debug(
                        f'Receiving aggregation strategy from {msg.sender_name}')
                    self.federated_aggregation_strategies[msg.sender_name] = msg.payload
                    logging.debug(
                        f'Reived aggregation strategy from {msg.sender_name}')

                    clients_registered = len(self.clients)
                    strategies_received = len(
                        self.federated_aggregation_strategies)
                    strategies_min = self.federated_aggregation_strategy.min_clients

                    if strategies_received == max(strategies_min, clients_registered):
                        self.send_to_all_servers(
                            comm_type='ToServerAggregationStrategies', payload=self.federated_aggregation_strategies)

                        self.broadcast(comm_type='TriggerAggregation')

                        self._federated_aggregation_strategy_history.append(
                            self.federated_aggregation_strategy)

                        self._federated_aggregation_strategies_history.append(
                            self.federated_aggregation_strategies)

                        self.federated_aggregation_strategies = dict()

                    else:
                        logging.info(
                            f'Waiting for {max(clients_registered, strategies_min)-strategies_received} more clients to send their trained models')

                elif self.is_server(msg.sender_name):
                    logging.debug(
                        f'Receiving aggregation strategy from {msg.sender_name}')

                    self.federated_aggregation_strategy: BaseStrategy = msg.payload

                    logging.debug(
                        f'Aggregation strategy from {msg.sender_name} was received')

                    if self.federated_aggregation_strategy.is_completed:
                        self.broadcast(
                            comm_type='Terminate',
                            payload={
                                'server': self.federated_aggregation_strategy,
                                'clients': self._federated_aggregation_strategies_history[-1]
                            }
                        )
                        self.log_activity(msg.sender_name, 'Terminate')
                        logging.info(
                            f'Training completed at {datetime.now().strftime("%H:%M:%S")}')

                    else:
                        self.send_to_all_clients(
                            comm_type='SendAggregationStrategy', payload=self.federated_aggregation_strategy)

                        self.broadcast(comm_type='TriggerTraining')

                self.log_activity(
                    msg.sender_name, 'ToOrchestratorAggregationStrategy')

            elif msg.comm_type == 'UnregisterClient':
                """
                From 1 Client
                To 1 Orchestator
                """
                if self.is_client(msg.sender_name):
                    logging.info(
                        f'Unregistering client {msg.sender_name}')
                    self.log_activity(
                        agent_name=msg.sender_name, activity='UnregisterClient')

                    del self.clients[msg.sender_name]
                else:
                    logging.warning(
                        f'Unknown agent {msg.sender_name} trying to unregister')

            elif msg.comm_type == 'UnregisterServer':
                """
                From 1 Server
                To 1 Orchestator
                """
                if self.is_server(msg.sender_name):
                    logging.info(
                        f'Unregistering server {msg.sender_name}')
                    self.log_activity(
                        agent_name=msg.sender_name, activity='UnregisterServer')

                    del self.servers[msg.sender_name]
                else:
                    logging.warning(
                        f'Unknown agent {msg.sender_name} trying to unregister')

            else:
                logging.warning(
                    f'Unknown message type {msg.comm_type} from {msg.sender_name}')

    def _register_client(self, msg: FederatedDataCommunication) -> None:
        """ Register a client

        Args:
            - msg: FederatedDataCommunication
                The message received from the client

        Returns:
            - None
        """
        if self.clients.get(msg.sender_name):
            logging.warning(
                f'Client {msg.sender_name} already registered')
            self._send_to_agent(
                agent_name=msg.sender_name, comm_type='RegisterClient', payload=False)
            return

        self.clients[msg.sender_name] = {
            'address': msg.sender_address,
            'port': msg.sender_port
        }

        self.log_activity(msg.sender_name, 'RegisterClient')
        logging.info(f'Client {msg.sender_name} registered')

        self._send_to_agent(agent_name=msg.sender_name,
                            comm_type='RegisterClient', payload=True)

        self.broadcast(
            comm_type='SyncRegisteredAgents', payload={'clients': self.clients, 'servers': self.servers})

        self._send_to_agent(
            agent_name=msg.sender_name, comm_type='SendTrainingConfiguration', payload=self.training_configuration)

        self._send_to_agent(agent_name=msg.sender_name,
                            comm_type='SendAggregationStrategy', payload=self.federated_aggregation_strategy)

        self._send_to_agent(
            agent_name=msg.sender_name, comm_type='TriggerTraining')

        self.log_activity(msg.sender_name, 'TriggerTraining')

    def _register_server(self, msg: FederatedDataCommunication) -> None:
        """ Register a server

        Args:
            - msg: FederatedDataCommunication
                The message received from the server

        Returns:
            - None
        """
        if self.servers.get(msg.sender_name):
            logging.warning(
                f'Server {msg.sender_name} already registered')
            self._send_to_agent(
                agent_name=msg.sender_name, comm_type='RegisterServer', payload=False)
            return

        self.servers[msg.sender_name] = {
            'address': msg.sender_address,
            'port': msg.sender_port
        }

        self.log_activity(msg.sender_name, 'RegisterServer')
        logging.info(f'Server {msg.sender_name} registered')

        self._send_to_agent(agent_name=msg.sender_name,
                            comm_type='RegisterServer', payload=True)

        self.broadcast(
            comm_type='SyncRegisteredAgents', payload={'clients': self.clients, 'servers': self.servers})

        self._send_to_agent(
            agent_name=msg.sender_name, comm_type='SendTrainingConfiguration', payload=self.training_configuration)

        self._send_to_agent(agent_name=msg.sender_name,
                            comm_type='SendAggregationStrategy', payload=self.federated_aggregation_strategy)

    def _receive_sync_status(self, msg: FederatedDataCommunication) -> None:
        """ Receive the status of a client or a server

        Args:
            - msg: FederatedDataCommunication
                The message received from the client or the server

        Returns:
            - None
        """
        if self.is_client(msg.sender_name):
            self.clients[msg.sender_name]['status'] = msg.payload
        elif self.is_server(msg.sender_name):
            self.servers[msg.sender_name]['status'] = msg.payload
        elif self.is_orchestrator(msg.sender_name):
            pass
        else:
            logging.warning(
                f'Agent {msg.sender_name} not registered')
            return

        logging.debug(
            f'Agent {msg.sender_name} status: {msg.payload}')

        self.log_activity(
            msg.sender_name, 'SyncStatus', details=msg.payload)
