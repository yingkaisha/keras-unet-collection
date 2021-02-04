
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left'):
    '''
    The encoder block of 2-d V-net.
    
    vnet_left(X, channel, res_num, activation='ReLU', pool=True, batch_norm=False, name='left')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    pool_size = 2

    X = encode_layer(X, channel, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm, name='{}_encode'.format(name))
    
    if pool is not False:
        X = CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_pre_conv'.format(name))

    X = Res_CONV_stack(X, X, channel, res_num=res_num, activation=activation, 
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    return X

def vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right'):
    '''
    The decoder block of 2-d V-net.
    
    vnet_right(X, X_list, channel, res_num, activation='ReLU', unpool=True, batch_norm=False, name='right')
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        stack_num: number of convolutional layers.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: name of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    X_skip = X
    
    X = concatenate([X,]+X_list, axis=-1, name='{}_concat'.format(name))
    
    X = Res_CONV_stack(X, X_skip, channel, res_num, activation=activation, 
                       batch_norm=batch_norm, name='{}_res_conv'.format(name))
    
    return X

def vnet_2d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet'):
    '''
    The base of 2-d V-net.
    
    vnet_2d_base(input_tensor, filter_num, res_num_ini=1, res_num_max=3, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, name='vnet')
    
    Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural 
    networks for volumetric medical image segmentation. In 2016 fourth international conference 
    on 3D vision (3DV) (pp. 565-571). IEEE.
    
    The Two-dimensional version is inspired by:
    https://github.com/FENGShuanglang/2D-Vnet-Keras
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling).
        res_num_max: the max number of convolutional layers within a residual block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras layers.
            
    Output
    ----------
        X: output tensor.
    
    * This is a modified version of V-net for 2-d inputw.
    * The original work supports `pool=False` only. 
      If pool is True, 'max', or 'ave', an additional conv2d layer will be applied. 
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
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        res_num_ini: number of convolutional layers of the first first residual block (before downsampling).
        res_num_max: the max number of convolutional layers within a residual block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.               
        name: prefix of the created keras layers.
            
    Output
    ----------
        model: a keras model. 
    
    * This is a modified version of V-net for 2-d inputw.
    * The original work supports `pool=False` only. 
      If pool is True, 'max', or 'ave', an additional conv2d layer will be applied. 
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

