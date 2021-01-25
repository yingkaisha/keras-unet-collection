
from __future__ import absolute_import

from keras_unet_collection.activations import GELU, Snake
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax


def stride_conv(X, channel, pool_size=2, 
                activation='ReLU', 
                batch_norm=False, name='stride_conv'):
    '''
    stride convolutional layer --> batch normalization --> Activation
    *Proposed to replace max- and average-pooling layers 
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        pool_size: number of stride
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    '''
    
    bias_flag = not batch_norm
    
    # linear convolution with strides
    X = Conv2D(channel, pool_size, strides=(pool_size, pool_size), 
                            padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)
    
    # batch normalization
    if batch_norm:
        X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
    
    # activation
    activation_func = eval(activation)
    X = activation_func(name='{}_activation'.format(name))(X)
    
    return X

def attention_gate(X, g, channel,  
                   activation='ReLU', 
                   attention='add', name='att'):
    '''
    Additive attention gate as in Oktay et al. 2018
    
    Input
    ----------
        X: input tensor, i.e., upsampled tensor)
        g: gated tensor. Downsampled for coarser level, and have not concatenated with X
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is ReLU
                    
        attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
    Output
    ----------
        X_att: output tensor
    
    '''
    activation_func = eval(activation)
    attention_func = eval(attention)
    
    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)
    
    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)
    
    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    
    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)
    
    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #
    
    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    
    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))
    
    return X_att

def CONV_stack(X, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layer:
    ----------
    (Convolutional layer --> batch normalization --> Activation)*stack_num
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of stacked Conv2D-BN-Activation layers
        dilation_rate: option of dilated convolution kernel 
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
        
    '''
    
    bias_flag = not batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        
    return X

def Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    '''
    Stacked convolutional layers with residual path
     
    Input
    ----------
        X: input tensor
        X_skip: the tensor that does go into the residual path (usually it is a copy of X)
        channel: number of convolution filters
        res_num: number of convolutional layers within the residual path
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
        
    Output
    ----------
        X: output tensor
        
    '''  
    X = CONV_stack(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name=name)

    X = add([X_skip, X], name='{}_add'.format(name))
    
    activation_func = eval(activation)
    X = activation_func(name='{}_add_activation'.format(name))(X)
    
    return X

def Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False, name='sep_conv'):
    '''
    Depthwise separable convolution with 
    (optional) dilated convolution kernel and batch normalization.
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of stacked depthwise-pointwise layers
        dilation_rate: option of dilated convolution kernel 
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm
    
    for i in range(stack_num):
        X = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, padding='same', 
                            use_bias=bias_flag, name='{}_{}_depthwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_depthwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_depthwise_activation'.format(name, i))(X)

        X = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_{}_pointwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_pointwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_pointwise_activation'.format(name, i))(X)
    
    return X

def ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp'):
    '''
    Atrous Spatial Pyramid Pooling (ASPP)
    
    ----------
    Wang, Y., Liang, B., Ding, M. and Li, J., 2019. Dense semantic labeling 
    with atrous spatial pyramid pooling and decoder for high-resolution remote 
    sensing imagery. Remote Sensing, 11(1), p.20.
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters 
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
        
    *dilation rates are assigned as 6, 9, 12.
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm

    shape_before = X.get_shape().as_list()
    b4 = GlobalAveragePooling2D(name='{}_avepool_b4'.format(name))(X)
    
    b4 = expand_dims(expand_dims(b4, 1), 1, name='{}_expdim_b4'.format(name))
    
    b4 = Conv2D(channel, 1, padding='same', use_bias=bias_flag, name='{}_conv_b4'.format(name))(b4)
    
    if batch_norm:
        b4 = BatchNormalization(name='{}_conv_b4_BN'.format(name))(b4)
        
    b4 = activation_func(name='{}_conv_b4_activation'.format(name))(b4)
    
    b4 = Lambda(lambda X: image.resize(X, shape_before[1:3], method='bilinear', align_corners=True), 
                name='{}_resize_b4'.format(name))(b4)
    
    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_conv_b0'.format(name))(X)
    
    if batch_norm:
        b0 = BatchNormalization(name='{}_conv_b0_BN'.format(name))(b0)
        
    b0 = activation_func(name='{}_conv_b0_activation'.format(name))(b0)

    b_r6 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=6, batch_norm=True, name='{}_sepconv_r6'.format(name))
    b_r9 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=9, batch_norm=True, name='{}_sepconv_r9'.format(name))
    b_r12 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=12, batch_norm=True, name='{}_sepconv_r12'.format(name))
    
    return concatenate([b4, b0, b_r6, b_r9, b_r12])

def CONV_output(X, n_labels, kernel_size=1, 
                activation='Softmax', 
                name='conv_output'):
    '''
    Convolutional layer with output activation
    
    Input
    ----------
        X: input tensor
        n_labels: number of classification labels (larger than two)
        kernel_size: size of 2-d convolution kernels. Default option is 1-by-1
        activation: one of the `tensorflow.keras.layers` interface. Default option is Softmax
                    if None is received, then linear activation is applied, that said, not activation.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
        
    '''
    
    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)
    
    if activation:
        
        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)
            
        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X





