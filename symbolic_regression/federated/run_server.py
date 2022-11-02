import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('--address', dest='address', type=str, help='The address')
parser.add_argument('--port', dest='port', type=int, help='The port')

args = parser.parse_args()

training_configuration = {
    'POPULATION_SIZE': 100,
    'GENERATIONS': 20,
    'TOURNAMENT_SIZE': 3,
    'checkpoint_frequency': 10,
    'parsimony': .8,
    'parsimony_decay': .85,
    'const_range': (0, 1),
    'constants_optimization': True,
    'constants_optimization_method': 'ADAM',
    'constants_optimization_conf': {
        'task':'binary:logistic', #or 'regression:wmse'
        'learning_rate': 1e-4,
        'batch_size': 64,
        'epochs': 50,
        'verbose': 0,
        'gradient_clip':False,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-7
        },
    'genetic_operators_frequency': {
        'crossover': 1,
        'randomize': 1,
        'mutation': 1,
        'insert_node': 1,
        'delete_node': 1,
        'mutate_leaf': 1,
        'mutate_operator': 1,
        'recalibrate': 1
        },
}

from symbolic_regression.federated.Server import FederatedSRServer
server = FederatedSRServer(args.name, args.address, int(args.port), training_configuration, 'avg')

server.open_registrations(min_clients=2)
server.open_features_alignment()
#server.send_training_configuration()
