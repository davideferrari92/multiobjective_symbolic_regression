import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('-a', '--address', dest='address', type=str, help='The address')
parser.add_argument('-p', '--port', dest='port', type=str, help='The port')

parser.add_argument('-sa', '--server-address', dest='server_address', type=str, help='The address')
parser.add_argument('-sp', '--server-port', dest='server_port', type=str, help='The port')

args = parser.parse_args()

from symbolic_regression.federated.Client import FederatedSRClient

client = FederatedSRClient(
    name=args.name,
    address=args.address,
    port=int(args.port),
    server_address=args.server_address,
    server_port=int(args.server_port),
    data_path=''
)

client.register()
#client.features_alignment()
#client.receive_training_configuration()