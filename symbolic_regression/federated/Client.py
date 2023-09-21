import logging
from multiprocessing.connection import Listener
from typing import Dict, List
import pandas as pd

from symbolic_regression.federated.Agent import FederatedAgent
from symbolic_regression.federated.Communication import \
    FederatedDataCommunication
from symbolic_regression.federated.strategies.BaseStrategy import BaseStrategy


class FederatedSRClient(FederatedAgent):

    def __init__(self, name: str, address: str, port: int, orchestrator_address: str, orchestrator_port: int) -> None:
        """ This class implement a Federated Symbolic Regression Client

        A client is an agent that implements the training side of the
        Federated Learning algorithm. It receives the training configuration from the
        orchestrator, trains the model and sends the model update back so that it can be
        aggregated by the server.
        The client is developed to support the following communications from the orchestrator:
        
            - RegisterClient
                return the confirmation by the orchestrator of sucessful registration
            - SendAggregationStrategy
                this message is sent by the orchestrator to server and clients to pass the aggregation strategy to be used
            - SendTrainingConfiguration
                this message is sent by the orchestrator to server and clients to pass the training configuration to be used
            - SyncRegisteredAgents
                this message is sent by the orchestrator to server and clients to sync the list of registered agents every time
                a new agent is registered or unregistered
            - Terminate
                this message is sent by the orchestrator to server and clients to trigger the termination of the server.
            - TriggerAggregation
                this message is sent by the orchestrator to server and clients to trigger the aggregation of the models
                It has effect only on the server. Clients use TriggerTraining
            - TriggerTraining
                this message is sent by the orchestrator to server and clients to trigger the training of the models.
                In case of the server, it has no effect.

        Args:
            - name: str
                Name of the client
            - address: str
                Address of the client
            - port: int
                Port of the client
            - orchestrator_address: str
                Address of the orchestrator
            - orchestrator_port: int
                Port of the orchestrator

        Returns:
            - None
        """
        super().__init__(name=name, address=address, port=port, orchestrator_address=orchestrator_address,
                         orchestrator_port=orchestrator_port)

        self.mode: str = 'client'

        self.data: pd.DataFrame = None
        self.val_data: pd.DataFrame = None

    def load_data(self, data_path: str):
        """ Load data from a file

        Args:
            - data_path: str
                Path to the data file

        Returns:
            - None
        """
        if data_path.endswith('csv'):
            self.data = pd.read_csv(data_path)

        elif data_path.endswith('pkl'):
            self.data = pd.read_pickle(data_path)

        elif data_path.endswith('xlsx') or data_path.endswith('xls'):
            self.data = pd.read_excel(data_path)

        else:
            raise RuntimeError(f"File format not supported: {data_path}")

    def register(self) -> bool:
        """ Register the client to the orchestrator and start the client.
        It will automatically start the client listener loop to wait for messages from the orchestrator.

        Returns:
            - None
        """
        self.send_to_orchestrator(
            comm_type='RegisterClient', payload=self.name)

        try:
            self.run_client()
        except KeyboardInterrupt:
            logging.info('Client stopped by user')
            self.send_to_orchestrator(
                comm_type='UnregisterClient', payload=self.name)
        except Exception as e:
            logging.error(e)
            self.status = f'error: {e}'
            self.send_to_orchestrator(
                comm_type='UnregisterClient', payload=self.name)

    def run_client(self) -> None:
        """ Run the client

        It will start the listener and wait for messages from the orchestrator

        Returns:
            - None
        """
        if not self.orchestrator_address or not self.orchestrator_port:
            raise AttributeError(
                'Orchestrator address and port not set for this server')

        listener = Listener((self.address, self.port))

        logging.info(
            f'Client {self.name} listening on {self.address}:{self.port}...')

        while True:
            """ Wait for a message from a client or a server

            For a better execution flow, we recommend to invoke any sender method
            as last instruction of each branch of the if statement.
            This will allow the listener to immediately be ready to receive
            the next message.
            """
            conn = listener.accept()
            msg: FederatedDataCommunication = conn.recv()

            if msg.comm_type == 'RegisterClient':
                """
                From 1 Orchestrator
                To None
                """
                if msg.payload is True:
                    self.is_registered = True
                    self.status = 'idle'
                    logging.info(f'Client {self.name} registered')
                else:
                    self.is_registered = False
                    logging.warning(f'Client {self.name} already registered')

            elif msg.comm_type == 'SendAggregationStrategy':
                """
                From 1 Orchestrator
                To None
                """
                self.federated_aggregation_strategy: BaseStrategy = msg.payload
                self.federated_aggregation_strategy.name = self.name
                logging.debug(
                    f'Aggregation strategy received.')

            elif msg.comm_type == 'SendTrainingConfiguration':
                """
                From 1 Orchestrator
                To None
                """
                self.training_configuration = msg.payload
                logging.debug(
                    f'Training configuration received.')

            elif msg.comm_type == 'SyncRegisteredAgents':
                """
                From 1 Orchestrator
                To None
                """
                self.clients = msg.payload['clients']
                self.servers = msg.payload['servers']

            elif msg.comm_type == 'Terminate':
                """
                From 1 Orchestrator
                To None
                """
                logging.debug(f'Termination requested')
                server_strategy: BaseStrategy = msg.payload['server']
                server_strategy.mode = 'client'  # Would be orchestrator otherwise and therefore wouldnt' execute anything
                
                client_strategies: List[BaseStrategy] = msg.payload['clients']

                logging.info(
                    f'Computing validation performance on {len(client_strategies)} clients strategies')

                server_strategy.on_validation(data=self.data, val_data=self.val_data)

                client_strategies: Dict[str, BaseStrategy]
                for client_strategy in client_strategies.values():
                    client_strategy.on_validation(data=self.data, val_data=self.val_data)

                self.send_to_orchestrator(
                    comm_type='TerminationValidation', payload={
                        'server': server_strategy,
                        'clients': client_strategies
                    }
                )
                self.status = 'terminated'
                logging.info(f'Final validation completed')
                return

            elif msg.comm_type == 'TriggerAggregation':
                """
                From 1 Orchestrator
                To None
                """
                logging.info(f'Server aggregation triggered')

            elif msg.comm_type == 'TriggerTraining':
                """
                From 1 Orchestrator
                To 1 Orchestrator
                """
                self.status = 'training'

                self.federated_aggregation_strategy.execute(data=self.data, val_data=self.val_data)

                self.send_to_orchestrator(
                    comm_type='ToOrchestratorAggregationStrategy', payload=self.federated_aggregation_strategy)

                self.status = 'trained_completed'

            else:
                logging.warning(f'Unknown message type: {msg.comm_type}')
