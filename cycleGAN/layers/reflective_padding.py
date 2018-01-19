from keras import backend as K

def reflective_padding_1(x):
    input_shape = x.shape
    left = K.reshape(x[:, :, 1, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    right = K.reshape(x[:, :, -2, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    upper = K.reshape(x[:, 1, :, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    lower = K.reshape(x[:, -2, :, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    
    x = K.concatenate([left, x, right], axis=2)
    x = K.concatenate([upper, x, lower], axis=1)
    return x

def reflective_padding_output_shape_1(input_shape):
    shape = list(input_shape)
    shape[1] += 1
    shape[2] += 1
    return tuple(shape)

def reflective_padding_3(x):
    input_shape = x.shape
    left = K.reshape(x[:, :, 1:4, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    right = K.reshape(x[:, :, -4:-1, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    upper = K.reshape(x[:, 1:4, :, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    lower = K.reshape(x[:, -4:-1, :, :], (input_shape[0], input_shape[1]+1, 1, input_shape[3]))
    
    x = K.concatenate([left, x, right], axis=2)
    x = K.concatenate([upper, x, lower], axis=1)
    return x

def reflective_padding_output_shape_3(input_shape):
    shape = list(input_shape)
    shape[1] += 3
    shape[2] += 3
    return tuple(shape)

# reflective padding層の実装　
# model.add(Lambda(reflective_padding, output_shape=reflective_padding_output_shape))