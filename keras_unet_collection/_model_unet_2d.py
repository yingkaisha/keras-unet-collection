
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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


def UNET_right(X, X_list, channel, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):
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
        concat: True for concatenating the corresponded X_list elements
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
        X = Conv2DTranspose(channel, pool_size, strides=(pool_size, pool_size), 
                                         padding='same', name='{}_trans_conv'.format(name))(X)
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X

def unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    
    '''
    The base of U-net with an optional ImageNet backbone.
    
    unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num: an iterable that defines the number of filters for each \
                      down- and upsampling level. E.g., [64, 128, 256, 512]
                      the depth is expected as `len(filter_num)`
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.
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
        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        X: the output tensor of the model base. 
    
    '''
    
    activation_func = eval(activation)

    X_skip = []
    depth_ = len(filter_num)

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                       batch_norm=batch_norm, name='{}_down0'.format(name))
        X_skip.append(X)

        # downsampling blocks
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                          batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
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

                X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool, 
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real+1))
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
        X = UNET_right(X, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_-1:
        for i in range(depth_-depth_decode-1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))   
    return X

def unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    '''
    U-net with an optional bakcbone
    
    unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_size: a tuple that defines the shape of input, e.g., (None, None, 3)
        filter_num: an iterable that defines the number of filters for each \
                      down- and upsampling level. E.g., [64, 128, 256, 512]
                      the depth is expected as `len(filter_num)`
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'.
                           Default option is Softmax
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
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
        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras model 
    
    '''
    activation_func = eval(activation)
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
        
    IN = Input(input_size)
    
    # base    
    X = unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, 
                     backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                     freeze_batch_norm=freeze_backbone, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model
