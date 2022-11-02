import logging
from multiprocessing.connection import Client, Listener

from symbolic_regression.federated.Communication import \
    FederatedDataCommunication


class FederatedSRServer():

    def __init__(self,
                 name: str, 
                 address: str, 
                 port: int, 
                 training_configuration: dict, 
                 strategy: str) -> None:
        """
        This class implement a Federated Symbolic Regression Server
        """
        self.name: str = name
        self.address: str = address
        self.port: int = port
        self.training_configuration: dict = training_configuration
        self.strategy: str = strategy

        self.registered_clients: dict = {}
        self.features_list: list = []

        self.communications: dict = {}

    def open_features_alignment(self):
        print(f'Server opened features alignment')
        logging.info(f'Server opened features alignment')

        features_lists = []

        listener = Listener((self.address, self.port))

        while len(features_lists) != len(self.registered_clients):
            conn = listener.accept()
            try:
                msg = conn.recv()
            except KeyboardInterrupt:
                logging.warning(
                    f'Features alignment stage interrupted via KeyboardInterrupt')
                return

            comm_object = msg['object']
            if not comm_object.comm_type == 'FeaturesAlignment':
                raise RuntimeError(
                    f'At this stage only FeaturesAlignment communications are allowed: {comm_object.comm_type}')
            features_lists.append(comm_object.payload)

        features_alignment_succeded = True

        # Checking that all features list are equal to all the others. May be optimised.

        for i in range(len(features_lists)):
            for j in range(len(features_lists)):
                if len(list(set(features_lists[i]) - set(features_lists[j]))) > 0:
                    logging.error(
                        f'The features lists are not the same from all clients')
                    features_alignment_succeded = False
                else:
                    self.features_list = features_lists[0]

        # Responding to all clients with the feature alignment outcome

        for fed_client, fed_client_data in self.registered_clients.items():
            conn = Client(
                (fed_client_data['address'], int(fed_client_data['port'])))
            conn.send({'object': FederatedDataCommunication(
                sender_name=self.name,
                sender_address=self.address,
                sender_port=self.port,
                comm_type='FeaturesAlignment',
                payload=features_alignment_succeded
            )})

            conn.close()

    def open_registrations(self, min_clients: int):
        logging.info(f'Server opened registrations')

        listener = Listener((self.address, self.port))

        while len(self.registered_clients) <= min_clients:
            conn = listener.accept()
            try:
                msg = conn.recv()
            except KeyboardInterrupt:
                logging.warning(
                    f'Registration stage interrupted via KeyboardInterrupt')
                return

            comm_object = msg['object']

            if self.registered_clients.get(comm_object.sender_name):
                logging.warning(
                    f'Client name already registered: {comm_object.sender_name}')

            self.registered_clients[comm_object.sender_name] = {
                'name': comm_object.sender_name,
                'address': comm_object.sender_address,
                'port': comm_object.sender_port
            }

            self._send_to_client(
                client_name=comm_object.sender_name,
                comm_type='RegistrationConfirmation',
                payload=None
            )

            logging.debug(f'Registered client: {comm_object}')

    def _send_to_client(self, client_name: str, comm_type: str, payload: object = None):
        conn = Client(
            (self.registered_clients[client_name]['address'], self.registered_clients[client_name]['port']))

        conn.send({'object': FederatedDataCommunication(
            sender_name=self.name,
            sender_address=self.address,
            sender_port=self.port,
            comm_type=comm_type,
            payload=payload
        )})

        conn.close()

    def _send_to_all_clients(self, comm_type: str, payload: object = None):
        for fed_client in self.registered_clients.keys():
            self._send_to_client(
                client_name=fed_client,
                comm_type=comm_type,
                payload=payload
            )

    def send_training_configuration(self):
        self._send_to_all_clients(
            comm_type='TrainingConfiguration',
            payload=self.training_configuration
        )
