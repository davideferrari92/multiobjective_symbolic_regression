import sys

from symbolic_regression.callbacks.CallbackSave import MOSRCallbackSaveCheckpoint
sys.path.append('/data/davide/workspace/multiobjective_symbolic_regression')

import argparse
import logging

from symbolic_regression.operators import *
from symbolic_regression.multiobjective.fitness.Classification import (
    AUC, BinaryCrossentropy, F1Score)
from symbolic_regression.federated.Orchestrator import Orchestrator
from symbolic_regression.federated.strategies.FedAvgNSGAII import FedAvgNSGAII


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('--address', dest='address', type=str, help='The address')
parser.add_argument('--port', dest='port', type=int, help='The port')

args = parser.parse_args()

name = 'fed_orchestrator'
address = 'localhost'
port = 5000

constants_optimization = 'sciipy'
constants_optimization_conf = {'task': 'binary:logistic'}

operations = [
    OPERATOR_ADD,
    OPERATOR_SUB,
    OPERATOR_MUL,
    OPERATOR_DIV,
    # OPERATOR_ABS,
    # OPERATOR_NEG,
    # OPERATOR_INV,
    OPERATOR_LOG,
    # OPERATOR_EXP,
    # OPERATOR_POW,
    OPERATOR_SQRT,
    OPERATOR_MAX,
    OPERATOR_MIN
]


configuration = {
    'federated': {
        'federated_rounds': 10,
        'max_rank_aggregation': 3,
        'min_clients': 3,
        'track_performance': True,
        'compatibility_check': True,
    },
    'symbolic_regressor': {
        'const_range': (0, 1),
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
        'parsimony': .8,
        'parsimony_decay': .85,
        'population_size': 100,
        'tournament_size': 3,
        'callbacks': [MOSRCallbackSaveCheckpoint(checkpoint_file='federatedSR_banknotes.checkpoint')],
    },
    'training': {
        'features': ['x1', 'x2', 'x3', 'x4'],
        'target': 'y',
        'weights': 'w',
        'operations': operations,
        'fitness_functions': [
            BinaryCrossentropy(label='bce', target='y', weights='w', logistic=True, constants_optimization=constants_optimization,
                               constants_optimization_conf=constants_optimization_conf, minimize=True, hypervolume_reference=1.1),
            AUC(label='1-auc', target='y', weights='w', logistic=True,
                one_minus=True, minimize=True, hypervolume_reference=1.1),
            F1Score(label='1-f1', target='y', weights='w', logistic=True, threshold=0.5,
                    one_minus=True, minimize=True, hypervolume_reference=1.1),
            AUC(label='auc', target='y', weights='w', logistic=True,
                one_minus=False, minimize=False, hypervolume_reference=1.1),
            F1Score(label='f1', target='y', weights='w', logistic=True, threshold=0.5,
                    one_minus=False, minimize=False, hypervolume_reference=1.1),
        ],
        'bootstrap_k': 100,
        'bootstrap_frac': 0.60,
        'constants_optimization': constants_optimization,
        'constants_optimization_conf': constants_optimization_conf,
        'generations_to_train': 5,
        'n_jobs': 10,
        'stop_at_convergence': False,
        'verbose': 2,
    }
}

logging.basicConfig(level=logging.DEBUG)

orchestrator = Orchestrator(
    name=name,
    address=address,
    port=port,
    training_configuration=configuration,
    save_path='federatedSR_banknotes.orchestrator'
)

orchestrator.federated_aggregation_strategy = FedAvgNSGAII(
    name='FedAvgNSGAII',
    mode='orchestrator',
    configuration=configuration,
)

orchestrator.register()
orchestrator.save()

print('Terminated')
