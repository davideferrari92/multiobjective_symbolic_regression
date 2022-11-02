class BaseCommunication:
    def __init__(self, sender_name: str, sender_address: str, sender_port: int, payload: object) -> None:
        self.sender_name = sender_name
        self.sender_address = sender_address
        self.sender_port = sender_port
        self.comm_type = None
        self.payload = payload


class FederatedDataCommunication(BaseCommunication):
    def __init__(self, sender_name: str, sender_address: str, sender_port: int, comm_type: str, payload: object) -> None:
        super().__init__(sender_name, sender_address, sender_port, payload)
        self.comm_type = comm_type
