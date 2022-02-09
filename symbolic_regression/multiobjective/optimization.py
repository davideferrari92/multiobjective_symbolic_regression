import tensorflow as tf
from symbolic_regression.Node import FeatureNode
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


def optimize(
        program,
        data,
        target,
        weights,
        constants_optimization_conf,
        task):
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
    if not isinstance(program.program, FeatureNode) and n_constants > 0 and n_features_used > 0:
        if program.const_range:
            const_range_min = program.const_range[0]
            const_range_max = program.const_range[1]
        else:
            const_range_min = -1
            const_range_max = 1

        from symbolic_regression.multiobjective.optimization import NNOptimizer

        constants_optimizer = NNOptimizer(
            units=1,
            n_features=n_features,
            n_constants=n_constants,
            const_range_min=const_range_min,
            const_range_max=const_range_max,
            exec_string=program.program.render(
                data=data.iloc[0, :], format_tf=True)
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
        model.compile(loss=loss, optimizer=opt, run_eagerly=False, metrics=['accuracy'])

        model.fit(
            data_tensor,
            target_tensor,
            sample_weight=weights_tensor,
            batch_size=constants_optimization_conf['batch_size'],
            epochs=constants_optimization_conf['epochs'],
            verbose=constants_optimization_conf['verbose']
        )

        final_parameters = list(model.get_weights()[0][0])

        program.set_constants(new=final_parameters)

    return program


class NNOptimizer(Layer):
    ''' 
    We create a single nuron NN with a customized activation function.
    '''

    def __init__(self,
                 units: int,
                 n_features: int,
                 n_constants: int,
                 const_range_min: float,
                 const_range_max: float,
                 exec_string: str
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
        self.const_range_min = const_range_min
        self.const_range_max = const_range_max

        self.exec_string = exec_string

    def build(self, input_shape):
        '''Create the state of the layer (weights)

        Weights are trainable parameters to be used as float values in symbolic regression after training
        - weights appear as a numpy array of shape (units,n_constants) 
        - weights are initially uniformly sampled in the range [const_range_min,const_range_max]
        - weights are set as trainable
        '''

        constants_init = tf.random_uniform_initializer(
            minval=self.const_range_min, maxval=self.const_range_max)
        self.constants = self.add_weight(name="constants", shape=(self.units, self.n_constants), dtype='float32',
                                         regularizer=None, initializer=constants_init, trainable=True)

        super().build(input_shape)

    def call(self, X):
        '''Defines the computation from inputs to outputs
        '''
        X = tf.split(X, self.n_features, 1)
        constants = tf.split(self.constants, self.n_constants, 1)

        return eval(self.exec_string)
