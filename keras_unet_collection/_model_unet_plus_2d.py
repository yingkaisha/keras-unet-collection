
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import warnings

def unet_plus_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                      activation='ReLU', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet'):
    '''
    The base of U-net++ with an optional ImageNet backbone
    
    unet_plus_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                      activation='ReLU', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet')
    
    ----------
    Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture 
    for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning 
    for Clinical Decision Support (pp. 3-11). Springer, Cham.
    
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
        deep_supervision: True for a model that supports deep supervision. Details see Zhou et al. (2018).
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
        If deep_supervision = False; Then the output is a tensor.
        If deep_supervision = True; Then the output is a list of tensors 
            with the first tensor obtained from the first downsampling level (for checking the input/output shapes only),
            the second to the `depth-1`-th tensors obtained from each intermediate upsampling levels (deep supervision tensors),
            and the last tensor obtained from the end of the base.
    
    '''
    
    activation_func = eval(activation)

    depth_ = len(filter_num)
    # allocate nested lists for collecting output tensors 
    X_nest_skip = [[] for _ in range(depth_)]

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # downsampling blocks (same as in 'unet_2d')
        X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                       batch_norm=batch_norm, name='{}_down0'.format(name))
        X_nest_skip[0].append(X)
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, 
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
            X_nest_skip[0].append(X)

    # backbone cases
    else:        
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_nest_skip[0] += backbone_([input_tensor,])
            depth_encode = len(X_nest_skip[0])

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_-1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_nest_skip[0] += backbone_([input_tensor,])
            depth_encode = len(X_nest_skip[0]) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor  
            X = X_nest_skip[0][-1]

            # extra downsamplings
            for i in range(depth_-depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool, 
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real+1))
                X_nest_skip[0].append(X)


    X = X_nest_skip[0][-1]

    for nest_lev in range(1, depth_):

        # depth difference between the deepest nest skip and the current upsampling  
        depth_lev = depth_-nest_lev

        # number of available encoded tensors
        depth_decode = len(X_nest_skip[nest_lev-1])

        # loop over individual upsamling levels
        for i in range(1, depth_decode):

            # collecting previous downsampling outputs
            previous_skip = []
            for previous_lev in range(nest_lev):
                previous_skip.append(X_nest_skip[previous_lev][i-1])

            # upsamping block that concatenates all available (same feature map size) down-/upsampling outputs
            X_nest_skip[nest_lev].append(
                UNET_right(X_nest_skip[nest_lev-1][i], previous_skip, filter_num[i-1], 
                           stack_num=stack_num_up, activation=activation, unpool=unpool, 
                           batch_norm=batch_norm, concat=False, name='{}_up{}_from{}'.format(name, nest_lev-1, i-1)))

        if depth_decode < depth_lev+1:

            X = X_nest_skip[nest_lev-1][-1]

            for j in range(depth_lev-depth_decode+1):
                j_real = j + depth_decode
                X = UNET_right(X, None, filter_num[j_real-1], 
                               stack_num=stack_num_up, activation=activation, unpool=unpool, 
                               batch_norm=batch_norm, concat=False, name='{}_up{}_from{}'.format(name, nest_lev-1, j_real-1))
                X_nest_skip[nest_lev].append(X)
            
    # output
    if deep_supervision:
        
        X_list = []
        
        for i in range(depth_):
            X_list.append(X_nest_skip[i][0])
        
        return X_list
        
    else:
        return X_nest_skip[-1][0]

def unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet'):
    '''
    U-net++ with an optional ImageNet backbone.
    
    unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet')
    
    ----------
    Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture 
    for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning 
    for Clinical Decision Support (pp. 3-11). Springer, Cham.
    
    Input
    ----------
        input_size: a tuple that defines the shape of input, e.g., (None, None, 3)
        filter_num: an iterable that defines the number of filters for each
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
        deep_supervision: True for a model that supports deep supervision. Details see Zhou et al. (2018).
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
    
    depth_ = len(filter_num)
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
    
    IN = Input(input_size)
    # base
    X = unet_plus_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                          activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, deep_supervision=deep_supervision, 
                          backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm, name=name)
    
    # output
    if deep_supervision:
        
        if (backbone is not None) and freeze_backbone:
            backbone_warn = '\n\nThe shallowest U-net++ deep supervision branch ("sup0") directly connects to a frozen backbone.\nTesting your configurations on `keras_unet_collection.base.unet_plus_2d_base` is recommended.'
            warnings.warn(backbone_warn);
            
        # model base returns a list of tensors
        X_list = X
        OUT_list = []
        
        print('----------\ndeep_supervision = True\nnames of output tensors are listed as follows (the last one is the final output):')
        
        # no backbone or VGG backbones
        # depth_ > 2 is expected (a least two downsampling blocks)
        if (backbone is None) or 'VGG' in backbone:
        
            for i in range(0, depth_-1):
                if output_activation is None:
                    print('\t{}_output_sup{}'.format(name, i))
                else:
                    print('\t{}_output_sup{}_activation'.format(name, i))
                    
                OUT_list.append(CONV_output(X_list[i], n_labels, kernel_size=1, activation=output_activation, 
                                            name='{}_output_sup{}'.format(name, i)))
        # other backbones        
        else:
            for i in range(1, depth_-1):
                if output_activation is None:
                    print('\t{}_output_sup{}'.format(name, i-1))
                else:
                    print('\t{}_output_sup{}_activation'.format(name, i-1))
                
                # an extra upsampling for creating full resolution feature maps
                if unpool:
                    X = UpSampling2D(size=(2, 2), name='{}_sup{}_unpool'.format(name, i-1))(X_list[i])
                else:
                    X = Conv2DTranspose(filter_num[i], 2, strides=(2, 2), padding='same', 
                                        name='{}_sup{}_trans_conv'.format(name, i-1))(X_list[i])
                    
                OUT_list.append(CONV_output(X, n_labels, kernel_size=1, activation=output_activation, 
                                        name='{}_output_sup{}'.format(name, i-1)))
                
        if output_activation is None:
            print('\t{}_output_final'.format(name))
        else:
            print('\t{}_output_final_activation'.format(name))
            
        OUT_list.append(CONV_output(X_list[-1], n_labels, kernel_size=1, activation=output_activation, name='{}_output_final'.format(name)))
        
    else:
        OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
        OUT_list = [OUT,]
        
    # model
    model = Model(inputs=[IN,], outputs=OUT_list, name='{}_model'.format(name))
    
    return model
