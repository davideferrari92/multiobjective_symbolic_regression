import tensorflow as tf
from tensorflow.keras.layers import Layer


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