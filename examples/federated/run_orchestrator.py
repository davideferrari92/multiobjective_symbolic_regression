import argparse
import sys

sys.path.append('../../')

from symbolic_regression.federated.Orchestrator import Orchestrator
from symbolic_regression.federated.strategies.FedNSGAII import \
    FedNSGAII
from symbolic_regression.multiobjective.fitness.Classification import (
    AUC, BinaryCrossentropy)
from symbolic_regression.multiobjective.fitness.Regression import NotConstant
from symbolic_regression.operators import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, help='The name')

parser.add_argument('--address', dest='address', type=str, help='The address')
parser.add_argument('--port', dest='port', type=int, help='The port')

args = parser.parse_args()

name = 'fed_orchestrator'
address = 'localhost'
port = 5000

constants_optimization = 'ADAM'
constants_optimization_conf = {
    'task': 'binary:logistic',  # or 'regression:wmse'
    'learning_rate': 1e-4,
    'batch_size': 64,
    'epochs': 50,
    'verbose': 0,
    'gradient_clip': False,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
    'l1_param': 1e-1,
    'l2_param': 0,
}

operations = [
    OPERATOR_ADD,
    OPERATOR_SUB,
    OPERATOR_MUL,
    OPERATOR_DIV,
    # OPERATOR_ABS,
    # OPERATOR_MOD,
    # OPERATOR_NEG,
    # OPERATOR_INV,
    OPERATOR_LOG,
    OPERATOR_EXP,
    OPERATOR_POW,
    OPERATOR_SQRT,
    # OPERATOR_MAX,
    # OPERATOR_MIN
]


configuration = {
    'federated': {
        'federated_rounds': 2,
        'min_clients': 2,
    },
    'symbolic_regressor': {
        'checkpoint_file': 'federatedSR_banknotes',
        'checkpoint_frequency': 1,
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
    },
    'training': {
        'features': ['x1', 'x2', 'x3', 'x4'],
        'operations': operations,
        'fitness_functions': [
            BinaryCrossentropy(label='bce', target='y', weights='w', logistic=True, constants_optimization=constants_optimization,
                               constants_optimization_conf=constants_optimization_conf, minimize=True, hypervolume_reference=1.1),
            AUC(label='1-auc', target='y', weights='w', logistic=True,
                one_minus=True, minimize=True, hypervolume_reference=1.1),
            NotConstant(label='not_constant', epsilon=.01,
                        minimize=True, hypervolume_reference=1.1)
        ],
        'generations_to_train': 2,
        'n_jobs': -1,
        'stop_at_convergence': False,
        'verbose': 2,
    }
}


orchestrator = Orchestrator(
    name=name,
    address=address,
    port=port,
    training_configuration=configuration,
    save_path='federatedSR_banknotes.orchestrator'
)

orchestrator.federated_aggregation_strategy = FedNSGAII(
    name='FedNSGAII',
    mode='orchestrator',
    configuration=configuration,
)

orchestrator.register()
orchestrator.save()

print('Terminated')
