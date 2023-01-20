import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append('../../')

from symbolic_regression.federated.Client import FederatedSRClient

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('-a', '--address', dest='address',
                    type=str, help='The address')
parser.add_argument('-p', '--port', dest='port', type=str, help='The port')

parser.add_argument('-sa', '--orchestrator-address',
                    dest='orchestrator_address', type=str, help='The address')
parser.add_argument('-sp', '--orchestrator-port',
                    dest='orchestrator_port', type=str, help='The port')

args = parser.parse_args()

name = 'fed_client2'
address = 'localhost'
port = 5003
orchestrator_address = 'localhost'
orchestrator_port = 5000

client = FederatedSRClient(
    name=name,
    address=address,
    port=port,
    orchestrator_address=orchestrator_address,
    orchestrator_port=orchestrator_port
)

data: pd.DataFrame = pd.read_csv('./examples/banknotes.csv').sample(500)
data['w'] = np.where(data['y'] == 1, 1./(2*data['y'].mean()),
                     1./(2*(1-data['y'].mean())))

client.data = data

client.register()
