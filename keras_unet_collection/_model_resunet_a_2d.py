
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def ResUNET_a_block(X, channel, kernel_size=3, dilation_num=1.0, activation='ReLU', batch_norm=False, name='res_a_block'):
    '''
    The "ResUNET-a" block
    
    ResUNET_a_block(X, channel, kernel_size=3, dilation_num=1.0, activation='ReLU', batch_norm=False, name='res_a_block')
    
    ----------
    Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for 
    semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114.
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        dilation_num: an iterable that defines dilation rates of convolutional layers.
                      stacks of conv2d is expected as `len(dilation_num)`.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    X_res = []
    
    for i, d in enumerate(dilation_num):
        
        X_res.append(CONV_stack(X, channel, kernel_size=kernel_size, stack_num=2, dilation_rate=d, 
                                activation=activation, batch_norm=batch_norm, name='{}_stack{}'.format(name, i)))
        
    if len(X_res) > 1:
        return add(X_res)
    
    else:
        return X_res[0]


def ResUNET_a_right(X, X_list, channel, kernel_size=3, dilation_num=[1,], 
                    activation='ReLU', unpool=True, batch_norm=False, name='right0'):
    '''
    The decoder block of ResUNet-a
    
    ResUNET_a_right(X, X_list, channel, kernel_size=3, dilation_num=[1,], 
                    activation='ReLU', unpool=True, batch_norm=False, name='right0')
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        dilation_num: an iterable that defines dilation rates of convolutional layers.
                      stacks of conv2d is expected as `len(dilation_num)`.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
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
        
    # <--- *stacked convolutional can be applied here
    X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = ResUNET_a_block(X, channel, kernel_size=kernel_size, dilation_num=dilation_num, activation=activation, 
                        batch_norm=batch_norm, name='{}_resblock'.format(name))
     
    return X

def resunet_a_2d_base(input_tensor, filter_num, dilation_num,
                      aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                      batch_norm=True, pool=True, unpool=True, name='resunet'):
    '''
    The base of ResUNet-a
    
    resunet_a_2d_base(input_tensor, filter_num, dilation_num,
                      aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                      batch_norm=True, pool=True, unpool=True, name='resunet')
                          
    ----------
    Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for 
    semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        dilation_num: an iterable that defines the dilation rates of convolutional layers.
                      Diakogiannis et al. (2020) suggested `[1, 3, 15, 31]`.
                      * This base function requires `len(filter_num) == len(dilation_num)`.
                      Explicitly defining dilation rates for each down-/upsampling level.
        aspp_num_down: number of Atrous Spatial Pyramid Pooling (ASPP) layer filters after the last downsampling block.
        aspp_num_up: number of ASPP layer filters after the last upsampling block.                 
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                   
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    * Downsampling is achieved through strided convolutional layers with 1-by-1 kernels in Diakogiannis et al., (2020), 
      and is here is achieved either with pooling layers or strided convolutional layers with 2-by-2 kernels.
    * If this base function is involved in network training, then the input shape cannot have NoneType.
    * `dilation_num` should be provided as 2d iterables, with the second dimension matches the model depth.
      e.g., for `len(filter_num) = 4`, dilation_num can be provided as: `[[1, 3, 15, 31], [1, 3, 15], [1,], [1,]]`.
      
    '''
    
    pool_size = 2
    
    activation_func = eval(activation)
    
    depth_ = len(filter_num)
    X_skip = []
    
    # ----- #
    # rejecting auto-mode from this base function
    if isinstance(dilation_num[0], int):
        raise ValueError('`resunet_a_2d_base` does not support automated determination of `dilation_num`.')
    else:
        dilation_ = dilation_num
    # ----- #
    
    X = input_tensor
    
    # input mapping with 1-by-1 conv
    X = Conv2D(filter_num[0], 1, 1, dilation_rate=1, padding='same', 
               use_bias=True, name='{}_input_mapping'.format(name))(X)
    X = activation_func(name='{}_input_activation'.format(name))(X)
    X_skip.append(X)
    # ----- #
    
    X = ResUNET_a_block(X, filter_num[0], kernel_size=3, dilation_num=dilation_[0], 
                        activation=activation, batch_norm=batch_norm, name='{}_res0'.format(name)) 
    X_skip.append(X)

    for i, f in enumerate(filter_num[1:]):
        ind_ = i+1
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_down{}'.format(name, i))
        #X = Conv2D(f, 1, 2, dilation_rate=1, padding='same', name='{}_down{}'.format(name, i))(X)
        #X = activation_func(name='{}_down{}_activation'.format(name, i))(X)

        X = ResUNET_a_block(X, f, kernel_size=3, dilation_num=dilation_[ind_], activation=activation, 
                            batch_norm=batch_norm, name='{}_resblock_{}'.format(name, ind_))
        X_skip.append(X)

    X = ASPP_conv(X, aspp_num_down, activation=activation, batch_norm=batch_norm, name='{}_aspp_bottom'.format(name))

    X_skip = X_skip[:-1][::-1]
    dilation_ = dilation_[:-1][::-1]
    
    for i, f in enumerate(filter_num[:-1][::-1]):

        X = ResUNET_a_right(X, [X_skip[i],], f, kernel_size=3, activation=activation, dilation_num=dilation_[i], 
                            unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    X = concatenate([X_skip[-1], X], name='{}_concat_out'.format(name))

    X = ASPP_conv(X, aspp_num_up, activation=activation, batch_norm=batch_norm, name='{}_aspp_out'.format(name))
    
    return X


def resunet_a_2d(input_size, filter_num, dilation_num, n_labels,
                 aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax', 
                 batch_norm=True, pool=True, unpool=True, name='resunet'):
    '''
    ResUNet-a
    
    resunet_a_2d(input_size, filter_num, dilation_num, n_labels,
                 aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax', 
                 batch_norm=True, pool=True, unpool=True, name='resunet')
                 
    ----------
    Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for 
    semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        dilation_num: an iterable that defines the dilation rates of convolutional layers.
                      Diakogiannis et al. (2020) suggested `[1, 3, 15, 31]`.
                      * `dilation_num` can be provided as 2d iterables, with the second dimension matches 
                      the model depth. e.g., for len(filter_num) = 4; dilation_num can be provided as: 
                      `[[1, 3, 15, 31], [1, 3, 15], [1,], [1,]]`.
                      * If `dilation_num` is not provided per down-/upsampling level, then the automated
                      determinations will be applied.
        n_labels: number of output labels.
        aspp_num_down: number of Atrous Spatial Pyramid Pooling (ASPP) layer filters after the last downsampling block.
        aspp_num_up: number of ASPP layer filters after the last upsampling block.  
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                  
        name: prefix of the created keras layers.
        
    Output
    ----------
        model: a keras model.
        
    * Downsampling is achieved through strided convolutional layers with 1-by-1 kernels in Diakogiannis et al., (2020), 
      and is here is achieved either with pooling layers or strided convolutional layers with 2-by-2 kernels.
    * `resunet_a_2d` does not support NoneType input shape.
    
    '''
    
    activation_func = eval(activation)
    depth_ = len(filter_num)
    
    X_skip = []
    
    # input_size cannot have None
    if input_size[0] is None or input_size[1] is None:
        raise ValueError('`resunet_a_2d` does not support NoneType input shape')
        
    # ----- #
    if isinstance(dilation_num[0], int):
        print("Received dilation rates: {}".format(dilation_num))
    
        deep_ = (depth_-2)//2
        dilation_ = [[] for _ in range(depth_)]
        
        print("Received dilation rates are not defined on a per downsampling level basis.")
        print("Automated determinations are applied with the following details:")
        
        for i in range(depth_):
            if i <= 1:
                dilation_[i] += dilation_num
            elif i > 1 and i <= deep_+1:
                dilation_[i] += dilation_num[:-1]
            else:
                dilation_[i] += [1,]
            print('\tdepth-{}, dilation_rate = {}'.format(i, dilation_[i]))
            
    else:
        dilation_ = dilation_num
    # ----- #
    
    IN = Input(input_size)
    
    # base
    X = resunet_a_2d_base(IN, filter_num, dilation_,
                          aspp_num_down=aspp_num_down, aspp_num_up=aspp_num_up, activation=activation,
                          batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))

    model = Model([IN], [OUT,])
    
    return model

