import logging
from multiprocessing.connection import Client, Listener

import pandas as pd
from symbolic_regression.SymbolicRegressor import SymbolicRegressor
from symbolic_regression.federated.Communication import FederatedDataCommunication


class FederatedSRClient():

    def __init__(self,
                 name: str,
                 address: str,
                 port: int,
                 server_address: str,
                 server_port: int,
                 data_path: str) -> None:
        """
        This class implement a Federated Symbolic Regression Client
        """

        self.name: str = name
        self.address: str = address
        self.port: int = port
        self.server_address: str = server_address
        self.server_port: int = server_port
        self.data_path: str = data_path

        self.data: pd.DataFrame = None

        self.is_registered: bool = False
        self.is_data_loaded: bool = False
        self.training_configuration: bool = None

        self.symbolic_regressor: SymbolicRegressor = None

    def features_alignment(self):
        # if not self.is_data_loaded:
        #    raise AttributeError(
        #        f'Data was never never loaded for this client')

        conn = Client((self.server_address, int(self.server_port)))

        conn.send({'object': FederatedDataCommunication(
            sender_name=self.name,
            sender_address=self.address,
            sender_port=self.port,
            comm_type='FeaturesAlignment',
            payload=[1, 2, 3]  # self.data.columns
        )})

        conn.close()

        listener = Listener((self.address, int(self.port)))

        conn = listener.accept()
        msg = conn.recv()
        comm_object = msg['object']

        if comm_object.payload == True:
            logging.info(f'Feature alignment succeded')
            self.is_features_aligned = True
        else:
            logging.error(f'Features alignment failed')
            self.is_features_aligned = False

    def load_data(self):
        if self.data_path.endswith('csv'):
            self.data = pd.read_csv(self.data_path)

        elif self.data_path.endswith('pkl'):
            self.data = pd.read_pickle(self.data_path)

        elif self.data_path.endswith('xlsx') or self.data_path.endswith('xls'):
            self.data = pd.read_excel(self.data_path)

        else:
            raise RuntimeError(f"File format not supported: {self.data_path}")

        self.is_data_loaded = True

    def receive_training_configuration(self):
        listener = Listener((self.address, self.port))

        conn = listener.accept()
        msg = conn.recv()
        comm_object = msg['object']

        if not comm_object.comm_type == 'TrainingConfiguration':
            raise RuntimeError(
                f'At this stage only FeaturesAlignment communications are allowed: {comm_object.comm_type}')

        self.training_configuration = comm_object.payload

    def send_dataset_metadata(self):
        pass

    def _send_to_server(self, comm_type: str, payload: object = None):
        conn = Client((self.server_address, int(self.server_port)))
        conn.send({'object': FederatedDataCommunication(
            sender_name=self.name,
            sender_address=self.address,
            sender_port=self.port,
            comm_type=comm_type,
            payload=payload
        )})

        conn.close()

    def register(self) -> bool:
        self._send_to_server(
            comm_type='FederatedRegistration',
            payload=None
        )

        listener = Listener((self.address, self.port))

        conn = listener.accept()
        msg = conn.recv()
        comm_object = msg['object']

        if not comm_object.comm_type == 'RegistrationConfirmation':
            raise RuntimeError(
                f'At this stage only RegistrationConfirmation communications are allowed: {comm_object.comm_type}')

        logging.info(f'Client is now registered')

        self.is_registered = True
