from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Lambda
from keras.models import Model
import layers

def generator(input_shape):
    
    def residual_block(x, out_dim):
        y = Lambda(reflective_padding_1, output_shape=reflective_padding_output_shape_1)(x)
        y = Conv2D(out_dim,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(y)
        y = InstanceNormalization()(y)
        
        y = Lambda(reflective_padding_1, output_shape=reflective_padding_output_shape_1)(y)
        y = Conv2D(out_dim,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(y)
        y = InstanceNormalization()(y)
        
        return y + x
    
    inp = Input(shape=input_shape)
    c0 = Lambda(reflective_padding_3, output_shape=reflective_padding_output_shape_3)(inp)
    
    c1 = Conv2D(64,
               kernel_size=7,
               strides=1,
               padding='valid',
               use_bias=False,
               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(c0)
    i1 = InstanceNormalization()(c1)
    a1 = Activation('relu')(i1)
    
    c2 = Conv2D(64*2,
               kernel_size=3,
               strides=2,
               padding='same',
               use_bias=False,
               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(a1)
    i2 = InstanceNormalization()(c2)
    a2 = Activation('relu')(i2)
    
    c3 = Conv2D(64*4,
               kernel_size=3,
               strides=2,
               padding='same',
               use_bias=False,
               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(a2)
    i3 = InstanceNormalization()(c3)
    a3 = Activation('relu')(i3)
    
    r1 = residual_block(a3, 64*4)
    r2 = residual_block(r1, 64*4)
    r3 = residual_block(r2, 64*4)
    r4 = residual_block(r3, 64*4)
    r5 = residual_block(r4, 64*4)
    r6 = residual_block(r5, 64*4)
    r7 = residual_block(r6, 64*4)
    r8 = residual_block(r7, 64*4)
    r9 = residual_block(r8, 64*4)
    
    d1 = Conv2DTranspose(64*2,
                        kernel_size=4,
                        strides=2,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(r9)
    di1 = InstanceNormalization()(d1)
    da1 = Activation('relu')(di1)
    d2 = Conv2DTranspose(64,
                        kernel_size=4,
                        strides=2,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(da1)
    di2 = InstanceNormalization()(d2)
    da2 = Activation('relu')(di2)
    dp2 = Lambda(reflective_padding_3, output_shape=reflective_padding_output_shape_3)(da2)
    dc3 = Conv2D(3,
               kernel_size=7,
               strides=1,
               padding='valid',
               use_bias=False,
               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(dp2)
    pred = Activation('tanh')(dc3)
    
    model = Model(inputs=inp, outputs=pred)
    
    return model