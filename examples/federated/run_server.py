import argparse
import sys

sys.path.append('/data/davide/workspace/multiobjective_symbolic_regression')
from symbolic_regression.federated.Server import FederatedSRServer

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('--address', dest='address', type=str, help='The address')
parser.add_argument('--port', dest='port', type=int, help='The port')

args = parser.parse_args()

name = 'fed_server'
address = 'localhost'
port = 5001
orchestrator_address = 'localhost'
orchestrator_port = 5000

import logging
logging.basicConfig(level=logging.DEBUG)

server = FederatedSRServer(
    name=name,
    address=address,
    port=port,
    orchestrator_address=orchestrator_address,
    orchestrator_port=orchestrator_port
)

server.register()
