""" EXPERIMENTAL: YOUR MILAGE MAY VARY!
This was a copy of the _model_unet_2d.py, but adapted to include 3d convolutions.

Note that you can use this method to predict 3d maps, but often was the case in my work that I wanted a 2d map at the end

Adapted by Randy J. Chase 

"""

from __future__ import absolute_import

from keras_unet_collection.layer_utils_3d import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker

from tensorflow.keras.layers import Input,Reshape
from tensorflow.keras.models import Model

def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', l1=1e-2, l2=1e-2,
              pool=True, batch_norm=False, kernel_initializer='glorot_uniform', name='left0',
              pool_size=(2,2,2)):
    '''
    The encoder block of U-net.
    
    UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        kernel_initializer: how initialize weights. 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    X = encode_layer(X, channel, pool_size, pool, activation=activation, l1=l1, l2=l2,
                     batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                     name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, l1=l1, l2=l2,
                   batch_norm=batch_norm, kernel_initializer=kernel_initializer, 
                   name='{}_conv'.format(name))
    
    return X


def UNET_right(X, X_list, channel, kernel_size=3, l1=1e-2, l2=1e-2, stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, kernel_initializer='glorot_uniform',
               name='right0',pool_size=(2,2,2)):
    
    '''
    The decoder block of U-net.
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        concat: True for concatenating the corresponded X_list elements.
        kernel_initializer: how initialize weights.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    X = decode_layer(X, channel, pool_size, unpool, l1=l1, l2=l2,
                     activation=activation, batch_norm=batch_norm,kernel_initializer=kernel_initializer,
                     name='{}_decode'.format(name))
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, l1=l1, l2=l2,
                   batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                   name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X,]+X_list, axis=4, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, l1=l1, l2=l2,
                   batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                   name=name+'_conv_after_concat')
    
    return X

def unet_3d_base(input_tensor, filter_num, kernel_size=3, stack_num_down=2, stack_num_up=2, 
                 l1=1e-2, l2=1e-2, activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, 
                 freeze_batch_norm=True, kernel_initializer='glorot_uniform',
                 name='unet',pool_size=(2,2,2)):
    
    '''
    The base of U-net with an optional ImageNet-trained backbone.
    
    unet_3d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        kernel_size: number of the size of the convolutional kernel within the convolutions.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        kernel_initializer: how initialize weights.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    activation_func = eval(activation)

    X_skip = []
    depth_ = len(filter_num)

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num[0], kernel_size=kernel_size, stack_num=stack_num_down, l1=l1, l2=l2,
                        activation=activation, batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                        name='{}_down0'.format(name))

        X_skip.append(X)

        # downsampling blocks
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, kernel_size=kernel_size,stack_num=stack_num_down, l1=l1, l2=l2,
                        activation=activation, pool=pool, batch_norm=batch_norm,kernel_initializer=kernel_initializer,
                        name='{}_down{}'.format(name, i+1),pool_size=pool_size)        
            X_skip.append(X)

    # backbone cases
    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor,])
            depth_encode = len(X_skip)
            
        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_-1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor,])
            depth_encode = len(X_skip) + 1


        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor  
            X = X_skip[-1]

            # extra downsamplings
            for i in range(depth_-depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num[i_real],kernel_size=kernel_size, stack_num=stack_num_down, 
                                l1=l1, l2=l2, activation=activation, pool=pool, 
                                batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                                name='{}_down{}'.format(name, i_real+1),pool_size=pool_size)
                X_skip.append(X)

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        X = UNET_right(X, [X_decode[i],], filter_num_decode[i], kernel_size=kernel_size, stack_num=stack_num_up, 
                        activation=activation, unpool=unpool, batch_norm=batch_norm, kernel_initializer=kernel_initializer,
                        l1=l1, l2=l2, name='{}_up{}'.format(name, i),pool_size=pool_size)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_-1:
        for i in range(depth_-depth_decode-1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real],kernel_size=kernel_size, stack_num=stack_num_up,
                            activation=activation, unpool=unpool, batch_norm=batch_norm, 
                            concat=False, kernel_initializer=kernel_initializer,
                            l1=l1, l2=l2, name='{}_up{}'.format(name, i_real),pool_size=pool_size)   
    return X

def unet_3d(input_size, filter_num, n_labels, kernel_size=3,stack_num_down=2, stack_num_up=2, l1=1e-2, l2=1e-2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, 
            kernel_initializer='glorot_uniform', name='unet',collapse=True,pool_size=(2,2,2)):
    '''
    U-net with an optional ImageNet-trained backbone.
    
    unet_3d(input_size, filter_num, n_labels, kernel_size=3, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        kernel_size: number of the size of the convolutional kernel within the convolutions.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
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
        kernel_initializer: how initialize weights.                 
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras model.
    
    '''
    activation_func = eval(activation)
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
        
    IN = Input(input_size)
    
    # base    
    X = unet_3d_base(IN, filter_num, kernel_size=kernel_size,stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, l1=l1, l2=l2,
                     backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                     freeze_batch_norm=freeze_backbone, kernel_initializer=kernel_initializer,
                     pool_size=pool_size,name=name)

    if collapse:
        #use valid padding to collapse third dim down and keep same number of filters 
        X = Conv3D(X.type_spec.shape[-1],kernel_size=(1, 1,input_size[2]),activation=activation,padding='valid',name='ConvolveAndCollapse')(X)

    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, 
                        l1=l1, l2=l2, kernel_initializer=kernel_initializer,name='{}_output'.format(name))
    
    if collapse:
        #get rid of 3rd dim where it is just a 1, orig shape should be [none,nx,ny,1,n_labels]
        OUT = Reshape([X.type_spec.shape[1],X.type_spec.shape[2],n_labels],name='squeeze')(OUT)

    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model
