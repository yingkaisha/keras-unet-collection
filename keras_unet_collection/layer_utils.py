
from __future__ import absolute_import

from keras_unet_collection.activations import GELU, Snake
from tensorflow import shape as tf_shape
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

def RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr'):
    '''
    Recurrent convolutional layers with skip connection.
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        
        stack_num: number of stacked recurrent convolutional layers
        recur_num: number of recurrent iterations.
        
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
        
    Output
    ----------
        X: output tensor
        
    '''
    
    activation_func = eval(activation)
    
    layer_skip = Conv2D(channel, 1, name='{}_conv'.format(name))(X)
    layer_main = layer_skip
    
    for i in range(stack_num):

        layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}'.format(name, i))(layer_main)
        
        if batch_norm:
            layer_res = BatchNormalization(name='{}_bn{}'.format(name, i))(layer_res)
        layer_res = activation_func(name='{}_activation{}'.format(name, i))(layer_res)
            
        for j in range(recur_num):
            
            layer_add = add([layer_res, layer_main], name='{}_add{}_{}'.format(name, i, j))
            layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}_{}'.format(name, i, j))(layer_add)
            
            if batch_norm:
                layer_res = BatchNormalization(name='{}_bn{}_{}'.format(name, i, j))(layer_res)
                
            layer_res = activation_func(name='{}_activation{}_{}'.format(name, i, j))(layer_res)
            
        layer_main = layer_res

    out_layer = add([layer_main, layer_skip], name='{}_add{}'.format(name, i))
    
    return out_layer

def ResUNET_a_block(X, channel, kernel_size=3, dilation_num=1.0, activation='ReLU', batch_norm=False, name='res_a_block'):
    '''
    ResUNET_a_block
    
    ----------
    Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for 
    semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114.
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        dilation_num: an iterable that defines dilation rates of convolutional layers
                      stacks of conv2d is expected as `len(dilation_num)`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    
    '''
    
    X_res = []
    
    for i, d in enumerate(dilation_num):
        
        X_res.append(CONV_stack(X, channel, kernel_size=kernel_size, stack_num=2, dilation_rate=d, 
                                activation=activation, batch_norm=batch_norm, name='{}_stack{}'.format(name, i)))
        
    if len(X_res) > 1:
        return add(X_res)
    
    else:
        return X_res[0]


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

def UNET_left(X, channel, kernel_size=3, 
              stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0'):
    '''
    Encoder block of UNet (downsampling --> stacked Conv2D)
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of convolutional layers
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        pool: True for maxpooling, False for strided convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    
    *downsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    '''
    pool_size = 2
    
    # maxpooling layer vs strided convolutional layers
    if pool:
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, batch_norm=batch_norm, name=name)
    
    # stack linear convolutional layers
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, batch_norm=batch_norm, name=name)
    
    return X

def UNET_RR_left(X, channel, kernel_size=3, 
                  stack_num=2, recur_num=2, activation='ReLU', 
                  pool=True, batch_norm=False, name='left0'):
    '''
    Encoder block of R2U-Net (downsampling --> RR CNN blocks)
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        
        stack_num: number of stacked recurrent convolutional layers
        recur_num: number of recurrent iterations.
        
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        pool: True for maxpooling, False for strided convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor
    
    *downsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    '''
    pool_size = 2
    
    # maxpooling layer vs strided convolutional layers
    if pool:
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, batch_norm=batch_norm, name=name)
    
    # stack linear convolutional layers
    X = RR_CONV(X, channel, stack_num=stack_num, recur_num=recur_num, 
                activation=activation, batch_norm=batch_norm, name=name)    
    return X

def RSU(X, channel_in, channel_out, depth=5, activation='ReLU', batch_norm=True, name='RSU'):
    '''
    Residual U-blocks (RSU)
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor
        channel_in: number of "intermediate" channels
        channel_out: number of "outside" channels
        depth: number of down- and upsampling levels
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    X_skip = []
    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, 
                   dilation_rate=1, activation=activation, batch_norm=batch_norm, 
                   name='{}_in'.format(name))
    X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_skip.append(X)
    
    for i in range(depth):
        X = MaxPooling2D(pool_size=(2, 2), name='{}_maxpool_{}'.format(name, i))(X)

        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))
        X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, 
               dilation_rate=2, activation=activation, batch_norm=batch_norm, 
               name='{}_up_0'.format(name))
    
    X_skip = X_skip[::-1]
    
    for i in range(depth):
        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i+1))
        
        X = UpSampling2D((2, 2), interpolation='bilinear', name='{}_unpool_{}'.format(name, i))(X)
  
    X = concatenate([X, X_skip[depth]], axis=-1, name='{}_concat_out'.format(name))

    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    X = add([X, X_skip[-1]], name='{}_out_add'.format(name))
    return X 

def RSU4F(X, channel_in, channel_out, dilation_num=[1, 2, 4, 8], activation='ReLU', batch_norm=True, name='RSU4F'):
    '''
    Residual U-blocks with dilated convolutiona kernels (RSU4F)
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor
        channel_in: number of "intermediate" channels
        channel_out: number of "outside" channels
        dilation_num: an iterable that defines dilation rates of convolutional layers.
                      Qin et al. (2020) suggested [1, 2, 4, 8].
                      
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    X_skip = []    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_in'.format(name))
    X_skip.append(X)
    
    for i, d in enumerate(dilation_num):

        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i))
        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    dilation_num = dilation_num[:-1][::-1]
    
    for i, d in enumerate(dilation_num[:-1]):

        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))

    X = concatenate([X, X_skip[2]], axis=-1, name='{}_concat_out'.format(name))
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    
    return add([X, X_skip[-1]], name='{}_out_add'.format(name))

def UNET_right(X, X_list, channel, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, name='right0'):
    '''
    Decoder block of UNet
    
    Input
    ----------
        X: input tensor
        X_list: a list of other tensors that connected to the input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of convolutional layers
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor

    *upsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    
    '''
    
    pool_size = 2
    
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), name='{}_unpool'.format(name))(X)
    else:
        # Transpose convolutional layer --> stacked linear convolutional layers
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                         padding='same', name='{}_trans_conv'.format(name))(X)
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    
    # <--- *stacked convolutional can be applied here
    X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X

def UNET_RR_right(X, X_list, channel, kernel_size=3, 
                   stack_num=2, recur_num=2, activation='ReLU',
                   unpool=True, batch_norm=False, name='right0'):
    '''
    Decoder block of R2U-Net
    
    Input
    ----------
        X: input tensor
        X_list: a list of other tensors that connected to the input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        
        stack_num: number of stacked recurrent convolutional layers
        recur_num: number of recurrent iterations.
        
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor

    *upsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    
    '''
    
    pool_size = 2
    
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), name='{}_unpool'.format(name))(X)
    else:
        # Transpose convolutional layer --> stacked linear convolutional layers
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                         padding='same', name='{}_trans_conv'.format(name))(X)
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    
    # Tensor concatenation
    H = concatenate([X,]+X_list, axis=-1, name='{}_concat'.format(name))
    
    # stacked linear convolutional layers after concatenation
    H = RR_CONV(H, channel, stack_num=stack_num, recur_num=recur_num, 
                      activation=activation, batch_norm=batch_norm, name=name)
    
    return H

def UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):
    '''
    Decoder block of Attention-UNet
    
    Input
    ----------
        X: input tensor
        X_left: the output of corresponded downsampling output tensor (the input tensor is upsampling input)
        channel: number of convolution filters
        att_channel: number of intermediate channel.        
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        atten_activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is ReLU
        attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
                   
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers
    Output
    ----------
        X: output tensor

    *upsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    
    '''
    
    pool_size = 2
    
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), name='{}_unpool'.format(name))(X)
    else:
        # Transpose convolutional layer --> stacked linear convolutional layers
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                         padding='same', name='{}_trans_conv'.format(name))(X)
        
    X_left = attention_gate(X=X_left, g=X, channel=att_channel, activation=atten_activation, 
                            attention=attention, name='{}_att'.format(name))
    
    # Tensor concatenation
    H = concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))
    
    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))
    
    return H

def ResUNET_a_right(X, X_list, channel, kernel_size=3, dilation_num=[1,], 
                    activation='ReLU', unpool=True, batch_norm=False, name='right0'):
    '''
    Decoder block of ResUNet-a
    
    Input
    ----------
        X: input tensor
        X_list: a list of other tensors that connected to the input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        dilation_num: an iterable that defines dilation rates of convolutional layers
                      stacks of conv2d is expected as `len(dilation_num)`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor.

    *upsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    
    '''
    
    pool_size = 2
    
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), name='{}_unpool'.format(name))(X)
    else:
        # Transpose convolutional layer --> stacked linear convolutional layers
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                         padding='same', name='{}_trans_conv'.format(name))(X)
        
    # <--- *stacked convolutional can be applied here
    X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = ResUNET_a_block(X, channel, kernel_size=kernel_size, dilation_num=dilation_num, activation=activation, 
                        batch_norm=batch_norm, name='{}_resblock'.format(name))
     
    return X

