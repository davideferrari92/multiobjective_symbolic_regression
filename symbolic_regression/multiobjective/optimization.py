from typing import Union
import warnings
import numpy as np
import pandas as pd
import sympy as sym
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
from symbolic_regression.Node import FeatureNode
from symbolic_regression.Program import Program
from sympy.utilities.lambdify import lambdify
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer

from symbolic_regression.multiobjective.utils import to_logistic

silence_tensorflow()
warnings.filterwarnings("ignore")


def optimize(program: Program,
             data: Union[dict, pd.Series, pd.DataFrame, None],
             target: str,
             weights: str,
             constants_optimization_conf: dict,
             constants_optimization_method: str = 'ADAM',
             task: str = 'regression:wmse'):
    """
    """
    if task not in ['regression:wmse', 'binary:logistic']:
        raise AttributeError(
            f'Task supported are regression:wmse or binary:logistic')

    n_constants = len(program.get_constants())
    n_features = len(program.features)
    n_features_used = len(program.features_used)
    '''
    not isinstance(program.program, FeatureNode)
        programs with only a FeatureNode are not acceptable anyway

    n_constants > 0
        as the optimization algorithm optimize only constants

    n_features_used > 0
        as it is a constant program anyway and the optimized won't work with this configuration
    '''
    if not isinstance(program.program,
                      FeatureNode) and n_constants > 0 and n_features_used > 0:
        if constants_optimization_method == 'SDG':
            f_opt = SGD
            prog = program
        if constants_optimization_method == 'ADAM':
            f_opt = ADAM
            prog = program
        if constants_optimization_method == 'NN':
            f_opt = NN
            if task=='binary:logistic':
                prog = to_logistic(program=program)

        final_parameters, _, _ = f_opt(
            program=prog,
            data=data,
            target=target,
            weights=weights,
            constants_optimization_conf=constants_optimization_conf,
            task=task
        )
        if len(final_parameters) > 0:
            prog.set_constants(new=final_parameters)
            
        if constants_optimization_method == 'NN':
            program.program = prog.program.operands[0]
            program.program.father = None

    return program


class NNOptimizer(Layer):
    ''' 
    We create a single nuron NN with a customized activation function.
    '''

    def __init__(self,
                 units: int,
                 n_features: int,
                 n_constants: int,
                 exec_string: str,
                 constants_list: list
                 ):
        '''Initializes the class and sets up the internal variables

        Args:
        - units:int = number of neurons in the NN (1 -> single neuron)
        - n_features:int = number of input variables (number of columns in dataset, i.e. number features) 
        - n_constants:int = number of numerical parameters appearing in program (number of float numbers used in program)
        - const_range_min:float = lower bound in float range
        - const_range_max:float = upper bound in float range
        '''
        super(NNOptimizer, self).__init__()
        self.units = units
        self.n_constants = n_constants
        self.n_features = n_features
        self.constants_list = constants_list

        self.exec_string = exec_string

    def build(self, input_shape):
        '''Create the state of the layer (weights)

        Weights are trainable parameters to be used as float values in symbolic regression after training
        - weights appear as a numpy array of shape (units,n_constants) 
        - weights are initially uniformly sampled in the range [const_range_min,const_range_max]
        - weights are set as trainable
        '''
        constants_init=to_NN_weights_init(self.constants_list)
        self.constants = self.add_weight(name="constants", shape=(self.units, self.n_constants), dtype='float32',
                                         regularizer=None, initializer=constants_init, trainable=True)

        super().build(input_shape)

    def call(self, X):
        '''Defines the computation from inputs to outputs
        '''
        X = tf.split(X, self.n_features, 1)
        constants = tf.split(self.constants, self.n_constants, 1)

        return eval(self.exec_string)


def NN(program: Program,
       data: Union[dict, pd.Series, pd.DataFrame],
       target: str,
       weights: str,
       constants_optimization_conf: dict,
       task: str):
    from symbolic_regression.multiobjective.optimization import NNOptimizer

    constants_list = [item.feature for item in program.get_constants()]
    n_constants = len(program.get_constants())
    n_features = len(program.features)

    constants_optimizer = NNOptimizer(
        units=1,
        n_features=n_features,
        n_constants=n_constants,
        exec_string=program.program.render(
            data=data.iloc[0, :], format_tf=True),
        constants_list=constants_list
    )

    data_tensor = tf.constant(
        data[program.features].to_numpy(), dtype=tf.float32)

    target_tensor = tf.constant(
        data[target].to_numpy(), dtype=tf.float32)
    if weights:
        weights_tensor = tf.constant(
            data[weights].to_numpy(), dtype=tf.float32)
    else:
        weights_tensor = tf.ones_like(target_tensor)

    inputs = Input(shape=[len(program.features)], name="Input")

    tf.keras.backend.clear_session()
    model = Model(inputs=inputs, outputs=constants_optimizer(inputs))

    # https://keras.io/api/losses/
    if task == 'regression:wmse':
        # https://keras.io/api/losses/regression_losses/#meansquarederror-class
        loss = tf.keras.losses.MeanSquaredError()
    elif task == 'binary:logistic':
        # https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
        loss = tf.keras.losses.BinaryCrossentropy()

    opt = tf.keras.optimizers.Adam(
        learning_rate=constants_optimization_conf['learning_rate'])
    model.compile(loss=loss, optimizer=opt,
                  run_eagerly=False, metrics=['accuracy'])

    model.fit(
        data_tensor,
        target_tensor,
        sample_weight=weights_tensor,
        batch_size=constants_optimization_conf['batch_size'],
        epochs=constants_optimization_conf['epochs'],
        verbose=constants_optimization_conf['verbose']
    )

    return list(model.get_weights()[0][0]), [], []


def SGD(program: Program,
        data: Union[dict, pd.Series, pd.DataFrame],
        target: str,
        weights: str,
        constants_optimization_conf: dict,
        task: str):
    '''
    Stochastic Gradient Descent with analytic derivatives
    '''
    learning_rate = constants_optimization_conf['learning_rate']
    batch_size = constants_optimization_conf['batch_size']
    epochs = constants_optimization_conf['epochs']
    gradient_clip = constants_optimization_conf['gradient_clip']

    if not program.is_valid:  # No constants in program
        return [], [], []

    n_features = len(program.features)
    constants = np.array([item.feature for item in program.get_constants()])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []

    # Initialize symbols for variables and constants
    x_sym = ''
    for f in program.features:
        x_sym += f'{f},'
    x_sym = sym.symbols(x_sym)
    c_sym = sym.symbols('c0:{}'.format(n_constants))

    # Initialize ground truth and data arrays
    y_true = np.reshape(data[target].to_numpy(), (data[target].shape[0], 1))
    X_data = data[program.features].to_numpy()
    if weights:
        w = np.reshape(data[weights].to_numpy(), (data[weights].shape[0], 1))
    else:
        w = np.ones_like(y_true)
    # convert program render into sympy formula (symplify?)
    p_sym = program.program.render(format_diff=True)

    # compute program analytic gradients with respect to the constants to be optimized
    grad = []
    for i in range(n_constants):
        grad.append(sym.diff(p_sym, f'c{i}'))

    # define gradient and program python functions from sympy object
    try:
        pyf_grad = lambdify([x_sym, c_sym], grad)
        pyf_prog = lambdify([x_sym, c_sym], p_sym)
    except KeyError:  # When the function doesn't have sense
        return [], [], []
    
    # Define batches
    n_batches = int(X_data.shape[0] / batch_size)
    X_batch = np.array_split(X_data, n_batches, 0)
    y_batch = np.array_split(y_true, n_batches, 0)
    w_batch = np.array_split(w, n_batches, 0)

    log, loss = [], []  # lists to store learning process

    # initialize variance
    var = 0.

    for _ in range(epochs):
        for i in range(n_batches):

            # Define current batch weights, and compute numerical values of pyf_grad pyf_prog
            y_pred = pyf_prog(tuple(np.split(X_batch[i], n_features, 1)),
                              tuple(constants))
            num_grad = pyf_grad(tuple(np.split(X_batch[i], n_features, 1)),
                                tuple(constants))

            if task == 'regression:wmse':
                av_loss = np.nanmean(w_batch[i] * (y_pred-y_batch[i])**2)
                av_grad = np.array([
                    np.nanmean( 2. * w_batch[i]  (y_pred-y_batch[i]) * g)
                    for g in num_grad
                ])

            elif task == 'binary:logistic':
                # compute average loss
                # w=np.where(y_batch[i]==1, 1./(2*y_batch[i].mean()),  1./(2*(1-y_batch[i].mean())))
                # av_loss=np.nanmean(-w*y_batch[i]*np.log(y_pred+1e-20)-w*(1.-y_batch[i])*np.log(1.-y_pred+1e-20))
                sigma = 1. / (1. + np.exp(-y_pred)
                              )  # numerical value of sigmoid(program)
                av_loss = np.nanmean(
                    -w_batch[i] *
                    (y_batch[i] * np.log(sigma + 1e-20) +
                     (1. - y_batch[i]) * np.log(1. - sigma + 1e-20)))
                # compute average gradients
                av_grad = np.array([
                    np.nanmean(w_batch[i] * (sigma - y_batch[i]) * g)
                    for g in num_grad
                ])

            # try with new constants if loss is nan
            if np.isnan(av_loss):
                var += 0.2
                constants = np.random.normal(0.0, var, constants.shape)

            norm_grad = np.linalg.norm(av_grad)
            if gradient_clip and (norm_grad > 1.):  # normalize gradients
                av_grad = av_grad / (norm_grad + 1e-20)

            # Updating constants
            constants -= learning_rate * av_grad

        log.append(list(constants))
        loss.append(av_loss)

    return constants, loss, log


def ADAM(program: Program,
         data: Union[dict, pd.Series, pd.DataFrame],
         target: str,
         weights: str,
         constants_optimization_conf: dict,
         task: str):
    '''
    ADAM with analytic derivatives
    beta_1: float = 0.9, 
    beta_2: float = 0.999, 
    epsilon: float = 1e-07,
    '''

    learning_rate = constants_optimization_conf['learning_rate']
    batch_size = constants_optimization_conf['batch_size']
    epochs = constants_optimization_conf['epochs']
    gradient_clip = constants_optimization_conf['gradient_clip']
    beta_1 = constants_optimization_conf['beta_1']
    beta_2 = constants_optimization_conf['beta_2']
    epsilon = constants_optimization_conf['epsilon']

    if not program.is_valid:  # No constants in program
        return [], [], []

    n_features = len(program.features)
    constants = np.array([item.feature for item in program.get_constants()])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []

    # Initialize symbols for variables and constants
    x_sym = ''
    for f in program.features:
        x_sym += f'{f},'
    x_sym = sym.symbols(x_sym)
    c_sym = sym.symbols('c0:{}'.format(n_constants))

    # Initialize ground truth and data arrays
    y_true = np.reshape(data[target].to_numpy(), (data[target].shape[0], 1))
    X_data = data[program.features].to_numpy()
    if weights:
        w = np.reshape(data[weights].to_numpy(), (data[weights].shape[0], 1))
    else:
        w = np.ones_like(y_true)

    # convert program render into sympy formula (symplify?)
    p_sym = program.program.render(format_diff=True)
    print(p_sym)
    print()

    # compute program analytic gradients with respect to the constants to be optimized
    grad = []
    for i in range(n_constants):
        grad.append(sym.diff(p_sym, f'c{i}'))

    # define gradient and program python functions from sympy object
    try:
        pyf_grad = lambdify([x_sym, c_sym], grad)
        pyf_prog = lambdify([x_sym, c_sym], p_sym)
    except KeyError:  # When the function doesn't have sense
        return [], [], []
    # Define batches
    n_batches = int(X_data.shape[0] / batch_size)
    X_batch = np.array_split(X_data, n_batches, 0)
    y_batch = np.array_split(y_true, n_batches, 0)
    w_batch = np.array_split(w, n_batches, 0)

    log, loss = [], []  # lists to store learning process

    # Initialize Adam variables
    m = 0
    v = 0
    t = 1
    var = 0

    for _ in range(epochs):
        for i in range(n_batches):
            
            # Define current batch weights, and compute numerical values of pyf_grad pyf_prog
            y_pred = pyf_prog(tuple(np.split(X_batch[i], n_features, 1)), tuple(constants))
            num_grad = pyf_grad(tuple(np.split(X_batch[i], n_features, 1)), tuple(constants))
            
            if task == 'regression:wmse':
                av_loss = np.nanmean(w_batch[i] * (y_pred-y_batch[i])**2)
                av_grad = np.array([
                    np.nanmean( 2. * w_batch[i]  (y_pred-y_batch[i]) * g)
                    for g in num_grad
                ])

            elif task == 'binary:logistic':
                # compute average loss
                #w=np.where(y_batch[i]==1, 1./(2*y_batch[i].mean()),  1./(2*(1-y_batch[i].mean())))
                # av_loss=np.nanmean(-w*y_batch[i]*np.log(y_pred+1e-20)-w*(1.-y_batch[i])*np.log(1.-y_pred+1e-20))
                sigma = 1. / (1. + np.exp(-y_pred)
                              )  # numerical value of sigmoid(program)
                av_loss = np.nanmean(
                    -w_batch[i] *
                    (y_batch[i] * np.log(sigma + 1e-20) +
                     (1. - y_batch[i]) * np.log(1. - sigma + 1e-20)))
                # compute average gradients
                av_grad = np.array([
                    np.nanmean(w_batch[i] * (sigma - y_batch[i]) * g)
                    for g in num_grad
                ])

            # try with new constants if loss is nan
            if np.isnan(av_loss):
                var += 0.2
                constants = np.random.normal(0.0, var, constants.shape)

            norm_grad = np.linalg.norm(av_grad)
            if gradient_clip and (norm_grad > 1.):  # normalize gradients
                av_grad = av_grad / (norm_grad + 1e-20)

            # Updating momentum variables
            m = beta_1 * m + (1 - beta_1) * av_grad
            v = beta_2 * v + (1 - beta_2) * np.power(av_grad, 2)
            m_hat = m / (1 - np.power(beta_1, t))
            v_hat = v / (1 - np.power(beta_2, t))
            t += 1

            # Update constants
            constants -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        log.append(list(constants))
        loss.append(av_loss)

    return constants, loss, log


def to_NN_weights_init(constants_list):
    def inititializer(shape, dtype=tf.float32):
        return tf.reshape(tf.constant(constants_list, dtype=dtype), (shape[0], shape[1]))
    return inititializer
