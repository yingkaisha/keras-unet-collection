
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input#, Conv2D
# from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply
# from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model


def vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left'):
    '''
    Encoder block of 2-d VNet
    
    vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left')
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        res_num: number of convolutional layers within the residual path
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        pool: True for maxpooling, False for strided convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    
    *downsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    '''
    if pool:
        X = MaxPooling2D(pool_size=(2, 2), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size=2, activation=activation, 
                        batch_norm=batch_norm, name='{}_down'.format(name))

    X = Res_CONV_stack(X, X, channel, res_num=res_num, activation=activation, 
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    return X

def vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right'):
    '''
    Decoder block of vnet 2d
    
    vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right')
    
    Input
    ----------
        X: input tensor
        X_list: a list of other tensors that connected to the input tensor
        channel: number of convolution filters
        stack_num: number of convolutional layers
        res_num: number of convolutional layers within the residual path
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor
    
    '''
    
    bias_flag = not batch_norm
    
    if unpool:
        X = UpSampling2D(size=(2, 2), name='{}_unpool'.format(name))(X)
    else:
        X = Conv2DTranspose(channel, 2, strides=(2, 2), padding='same', use_bias=bias_flag, 
                            name='{}_trans_conv'.format(name))(X)
        if batch_norm:
            X = BatchNormalization(name='{}_up_bn'.format(name))(X)

        activation_func = eval(activation)
        X = activation_func(name='{}_up_activation'.format(name))(X)
    
    X_skip = X
    
    X = concatenate([X,]+X_list, axis=-1, name='{}_concat'.format(name))
    
    X = Res_CONV_stack(X, X_skip, channel, res_num, activation=activation, 
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    
    return X

def vnet_2d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet'):
    '''
    The base layers of vnet 2d
    
    vnet_2d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet')
    
    Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural 
    networks for volumetric medical image segmentation. In 2016 fourth international conference 
    on 3D vision (3DV) (pp. 565-571). IEEE.
    
    The Two-dimensional version is inspired by:
    https://github.com/FENGShuanglang/2D-Vnet-Keras
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num: an iterable that defines the number of filters for each \
                    down- and upsampling level. E.g., [64, 128, 256, 512]
                    the depth is expected as `len(filter_num)`   
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling)
        res_num_max: the max number of convolutional layers within a residual block
        
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
        name: prefix of the created keras layers.
            
    Output
    ----------
        X: the output tensor of the model base.
    
    * This is a modified version of V-net for 2-d input.
    * All the 5-by-5 convolutional kernels are changed (and fixed) to 3-by-3.
    '''

    depth_ = len(filter_num)

    # determine the number of res conv layers in each down- and upsampling level
    res_num_list = []
    for i in range(depth_):
        temp_num = res_num_ini + i
        if temp_num > res_num_max:
            temp_num = res_num_max
        res_num_list.append(temp_num)

    X_skip = []

    X = input_tensor
    # ini conv layer
    X = CONV_stack(X, filter_num[0], kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_input_conv'.format(name))

    X = Res_CONV_stack(X, X, filter_num[0], res_num=res_num_list[0], activation=activation, 
                 batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_skip.append(X)

    # downsampling levels
    for i, f in enumerate(filter_num[1:]):
        X = vnet_left(X, f, res_num=res_num_list[i+1], activation=activation, pool=pool, 
                      batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))

        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    filter_num = filter_num[:-1][::-1]
    res_num_list = res_num_list[:-1][::-1]

    # upsampling levels
    for i, f in enumerate(filter_num):
        X = vnet_right(X, [X_skip[i],], f, res_num=res_num_list[i], 
                       activation=activation, unpool=unpool, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))

    return X


def vnet_2d(input_size, filter_num, n_labels,
            res_num_ini=1, res_num_max=3, 
            activation='ReLU', output_activation='Softmax', 
            batch_norm=False, pool=True, unpool=True, name='vnet'):
    '''
    vnet 2d
    
    vnet_2d(input_size, filter_num, n_labels,
            res_num_ini=1, res_num_max=3, 
            activation='ReLU', output_activation='Softmax', 
            batch_norm=False, pool=True, unpool=True, name='vnet')
    
    Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural 
    networks for volumetric medical image segmentation. In 2016 fourth international conference 
    on 3D vision (3DV) (pp. 565-571). IEEE.
    
    The Two-dimensional version is inspired by:
    https://github.com/FENGShuanglang/2D-Vnet-Keras
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num: an iterable that defines the number of filters for each \
                    down- and upsampling level. E.g., [64, 128, 256, 512]
                    the depth is expected as `len(filter_num)`
        n_labels: number of output labels.
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling)
        res_num_max: the max number of convolutional layers within a residual block
        
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'.
                           Default option is Softmax
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
        name: prefix of the created keras layers.
            
    Output
    ----------
        model: a keras model 
    
    * This is a modified version of V-net for 2-d input.
    * All the 5-by-5 convolutional kernels are changed (and fixed) to 3-by-3.
    '''
    
    IN = Input(input_size)
    X = IN
    # base
    X = vnet_2d_base(X, filter_num, res_num_ini=res_num_ini, res_num_max= res_num_max, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model

