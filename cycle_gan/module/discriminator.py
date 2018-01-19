from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Sequential
import layers 

def discriminator(input_shape):
    
    model = Sequential()
    model.add(Conv2D(64,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)), input_shape=input_shape)
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(InstanceNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1,  
                     kernel_size=1,
                     strides=1,
                     use_bias=False,
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
    
    return model