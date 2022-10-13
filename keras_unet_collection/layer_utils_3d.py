""" EXPERIMENTAL: YOUR MILAGE MAY VARY!
This was a copy of the layer_utils.py, but adapted to include 3d convolutions.
    Adapted by Randy J. Chase 
"""
from __future__ import absolute_import

from keras_unet_collection.activations import GELU, Snake
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, UpSampling3D, Conv3DTranspose, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
from tensorflow.keras import regularizers

def decode_layer(X, channel, pool_size, unpool, kernel_size=3, l1=1e-2, l2=1e-2,
                 activation='ReLU', batch_norm=False,kernel_initializer='glorot_uniform',
                 name='decode'):
    '''
    An overall decode layer, based on either upsampling or trans conv.
    
    decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.           
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.
    
    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm
    
    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'
    
    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool keyword')
        
    if unpool:
        #will need to change this if not wanting to upsample 3rd dim. RJC 10/04/22
        X = UpSampling3D(size=pool_size, name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            #probably doesnt work here... RJC 10/4/22
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size, pool_size), 
                            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                            padding='same', kernel_initializer=kernel_initializer,
                            name='{}_trans_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
        
    return X

def encode_layer(X, channel, pool_size, pool, kernel_size='auto', l1=1e-2, l2=1e-2,
                 activation='ReLU', batch_norm=False, kernel_initializer='glorot_uniform',
                 name='encode'):
    '''
    An overall encode layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv2d.
    
    encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')
        
    # maxpooling2d as default
    if pool is True:
        pool = 'max'
        
    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm
    
    if pool == 'max':
        X = MaxPooling3D(pool_size=pool_size, name='{}_maxpool'.format(name))(X)
        
    elif pool == 'ave':
        X = AveragePooling3D(pool_size=pool_size, name='{}_avepool'.format(name))(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides !Probably doesnt work RJC 10/4/22
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size,pool_size), 
                   kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                   padding='valid', use_bias=bias_flag,kernel_initializer=kernel_initializer,
                   name='{}_stride_conv'.format(name))(X)
        
        
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

def CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, l1=1e-2, l2=1e-2, activation='ReLU',batch_norm=False, 
                    kernel_initializer='glorot_uniform',name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num
    
    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack')
    
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        kernel_initializer: how to initialize kernels. By defualt uses the default Conv. init. 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    bias_flag = not batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv3D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), 
                   dilation_rate=dilation_rate, kernel_initializer=kernel_initializer,
                   name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=4, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        
    return X

def CONV_output(X, n_labels, kernel_size=1, l1=1e-2, l2=1e-2, activation='Softmax',kernel_initializer='glorot_uniform',
                    name='conv_output'):
    '''
    Convolutional layer with output activation.
    
    CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output')
    
    Input
    ----------
        X: input tensor.
        n_labels: number of classification label(s).
        kernel_size: size of 2-d convolution kernels. Default is 1-by-1.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                    Default option is 'Softmax'.
                    if None is received, then linear activation is applied.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    X = Conv3D(n_labels, kernel_size, padding='same', use_bias=True, kernel_initializer=kernel_initializer,
                    kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), name=name)(X)
    
    if activation:
        
        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)
            
        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

