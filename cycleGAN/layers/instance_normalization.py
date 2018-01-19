from keras import backend as K
from keras.engine.topology import Layer

class InstanceNormalization(Layer):
    
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.scale = K.add_weight(name='scale',
                                 shape=input_shape[3],
                                 initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.02, seed=None),
                                 trainable=True)
        self.offset = K.add_weight(name='offset',
                                   shape=input_shape[3],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True)
        super(InstanceNormalization, self).build(input_shape) 
    
    def call(self, x):
        mean = K.mean(x, axis=[1, 2], keepdims=True)
        var = K.var(x, axis=[1, 2], keepdims=True)
        epsilon = K.epsilon(1e-05)
        
        inv = K.sqrt(var + epsilon)
        normalized = (input - mean) / inv
        return self.scale*normalized + self.offset
    
    def compute_output_shape(self, input_shape):
        return (input_shape)