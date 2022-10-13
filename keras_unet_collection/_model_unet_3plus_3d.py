""" EXPERIMENTAL: YOUR MILAGE MAY VARY!
This was a copy of the _model_unet_3plus_2d.py, but adapted to include 3d convolutions.

Note that you can use this method to predict 3d maps, but often was the case in my work that I wanted a 2d map at the end

Adapted by Randy J. Chase 

"""

from __future__ import absolute_import

from keras_unet_collection.layer_utils_3d import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker
from keras_unet_collection._model_unet_3d import UNET_left, UNET_right

from tensorflow.keras.layers import Input,Reshape
from tensorflow.keras.models import Model

import warnings

import numpy as np

def unet_3plus_3d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, kernel_size=3, l1=1e-2, l2=1e-2,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus',
                       pool_size=(2,2,2)):
    '''
    The base of UNET 3+ with an optional ImagNet-trained backbone.
    
    unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')
                  
    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.        
        filter_num_down: a list that defines the number of filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        kernel_size: number of the size of the convolutional kernel within the convolutions.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.          
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU                
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.     
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

    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.
    
    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with the first tensor has the smallest size. 
    
    '''
    
    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=kernel_size, stack_num=stack_num_down, l1=l1, l2=l2,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):

            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=kernel_size, stack_num=stack_num_down, activation=activation, 
                          l1=l1, l2=l2, pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1),
                          pool_size=pool_size)
            X_encoder.append(X)

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor,])
            depth_encode = len(X_encoder)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_-1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor,])
            depth_encode = len(X_encoder) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor  
            X = X_encoder[-1]

            # extra downsamplings
            for i in range(depth_-depth_encode):

                i_real = i + depth_encode

                X = UNET_left(X, filter_num_down[i_real], stack_num=stack_num_down, activation=activation, pool=pool, 
                              l1=l1, l2=l2, batch_norm=batch_norm, name='{}_down{}'.format(name, i_real+1),
                              pool_size=pool_size)
                X_encoder.append(X)


    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])

    # upsampling levels
    X_encoder = X_encoder[::-1]

    depth_decode = len(X_encoder)-1

    # loop over upsampling levels
    for i in range(depth_decode):

        f = filter_num_skip[i]

        # collecting tensors for layer fusion
        X_fscale = []

        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_decode):

            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev-i-1 # -1 for python indexing

            # deeper tensors are obtained from **decoder** outputs
            if pool_scale < 0:
                pool_size = 2**(-1*pool_scale)
                pool_size = (pool_size,pool_size,pool_size)

                X = decode_layer(X_decoder[lev], f, pool_size, unpool, l1=l1, l2=l2,
                     activation=activation, batch_norm=batch_norm, name='{}_up_{}_en{}'.format(name, i, lev))

            # unet skip connection (identity mapping)    
            elif pool_scale == 0:

                X = X_encoder[lev]

            # shallower tensors are obtained from **encoder** outputs
            else:
                pool_size = 2**(pool_scale)
                pool_size = (pool_size,pool_size,pool_size)

                X = encode_layer(X_encoder[lev], f, pool_size, pool, activation=activation, l1=l1, l2=l2,
                                 batch_norm=batch_norm, name='{}_down_{}_en{}'.format(name, i, lev))

            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=kernel_size, stack_num=1, l1=l1, l2=l2,
                           activation=activation, batch_norm=batch_norm, name='{}_down_from{}_to{}'.format(name, i, lev))

            X_fscale.append(X)  

        # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True

        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, filter_num_aggregate, kernel_size=kernel_size, stack_num=stack_num_up, l1=l1, l2=l2,
                       activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        X_decoder.append(X)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_-1:
        for i in range(depth_-depth_decode-1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_aggregate, stack_num=stack_num_up, activation=activation, l1=l1, l2=l2,
                           unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_plain_up{}'.format(name, i_real),
                           pool_size=pool_size)
            X_decoder.append(X)
        
    # return decoder outputs
    return X_decoder

def unet_3plus_3d(input_size, filter_num_down, n_labels, kernel_size=3, filter_num_skip='auto', filter_num_aggregate='auto', 
                  l1=1e-2, l2=1e-2, stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus',
                  collapse=True,pool_size=(2,2,2)):
    
    '''
    UNET 3+ with an optional ImageNet-trained backbone.
    
    unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')
                  
    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_down: a list that defines the number of filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        kernel_size: number of the size of the convolutional kernel within the convolutions.
        l1: the l1 regularization penalty used in kernel regularization
        l2: the l2 regularization penalty used in kernel regularization
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'
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
        deep_supervision: True for a model that supports deep supervision. Details see Huang et al. (2020).
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
        
    * The Classification-guided Module (CGM) is not implemented. 
      See https://github.com/yingkaisha/keras-unet-collection/tree/main/examples for a relevant example.
    * Automated mode is applied for determining `filter_num_skip`, `filter_num_aggregate`.
    * The default output activation is sigmoid, consistent with Huang et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.
    
    Output
    ----------
        model: a keras model.
    
    '''

    depth_ = len(filter_num_down)

    #check minimum depth:
    if depth_ < 3:
        print('WARNING: in order to setup all skip connections, a UNET3+ must have a minimum')
        print('depth of 3. You gave a depth of: {}. To change this, change the len of your filter_num_down list.'.format(depth_))

    #check input size for power of 2
    if (np.log2(input_size[0]).is_integer()) and (np.log2(input_size[1]).is_integer()):
        pass
    else:
        print('WARNING: At least one of your input shapes are not a power of two.')
        print('This might make things weird with maxpooling and concatenating the skip connections.')
        print('Best to make your data to have power of 2s [e.g., 32, 64, 128, 256, 512]')
        print('Your given input shape: ', input_size)
    
    verbose = False
    
    if filter_num_skip == 'auto':
        verbose = True
        filter_num_skip = [filter_num_down[0] for num in range(depth_-1)]
        
    if filter_num_aggregate == 'auto':
        verbose = True
        filter_num_aggregate = int(depth_*filter_num_down[0])
        
    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of convolution filters after each full-scale skip connection: filter_num_skip = {}'.format(filter_num_skip))
        print('\tNumber of channels of full-scale aggregated feature maps: filter_num_aggregate = {}'.format(filter_num_aggregate))    
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
    
    X_encoder = []
    X_decoder = []


    IN = Input(input_size)

    X_decoder = unet_3plus_3d_base(IN, filter_num_down, filter_num_skip, filter_num_aggregate, kernel_size=kernel_size,
                                   stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation, 
                                   batch_norm=batch_norm, pool=pool, unpool=unpool, l1=l1, l2=l2,
                                   backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                                   freeze_batch_norm=freeze_batch_norm, name=name,pool_size=(2,2,2))
    X_decoder = X_decoder[::-1]
    
    if deep_supervision:
        
        # ----- frozen backbone issue checker ----- #
        if ('{}_backbone_'.format(backbone) in X_decoder[0].name) and freeze_backbone:
            
            backbone_warn = '\n\nThe deepest UNET 3+ deep supervision branch directly connects to a frozen backbone.\nTesting your configurations on `keras_unet_collection.base.unet_plus_2d_base` is recommended.'
            warnings.warn(backbone_warn);
        # ----------------------------------------- #
        
        OUT_stack = []
        L_out = len(X_decoder)
        
        print('----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n')
        
        # conv2d --> upsampling --> output activation.
        # index 0 is final output 
        for i in range(1, L_out):
            
            pool_size = 2**(i)
            
            X = Conv2D(n_labels, kernel_size, l1=l1, l2=l2, padding='same', 
                            name='{}_output_conv_{}'.format(name, i-1))(X_decoder[i])
            
            X = decode_layer(X, n_labels, pool_size, unpool, l1=l1, l2=l2,
                             activation=None, batch_norm=False, name='{}_output_sup{}'.format(name, i-1))
            
            if output_activation:
                print('\t{}_output_sup{}_activation'.format(name, i-1))
                
                if output_activation == 'Sigmoid':
                    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
                else:
                    activation_func = eval(output_activation)
                    X = activation_func(name='{}_output_sup{}_activation'.format(name, i-1))(X)
            else:
                if unpool is False:
                    print('\t{}_output_sup{}_trans_conv'.format(name, i-1))
                else:
                    print('\t{}_output_sup{}_unpool'.format(name, i-1))
                    
            OUT_stack.append(X)
        
        X = CONV_output(X_decoder[0], n_labels, kernel_size=3, l1=l1, l2=l2,
                        activation=output_activation, name='{}_output_final'.format(name))
        OUT_stack.append(X)
        
        if output_activation:
            print('\t{}_output_final_activation'.format(name))
        else:
            print('\t{}_output_final'.format(name))
            
        model = Model([IN,], OUT_stack)

    else:
        X = X_decoder[0]
        if collapse:
            
            #use valid padding to collapse third dim down and keep same number of filters 
            X = Conv3D(X.type_spec.shape[-1],kernel_size=(1, 1,input_size[2]),activation=activation,padding='valid',name='ConvolveAndCollapse')(X)

        OUT = CONV_output(X, n_labels, kernel_size=3, l1=l1, l2=l2,
                          activation=output_activation, name='{}_output_final'.format(name))

        if collapse:
            #get rid of 3rd dim where it is just a 1, orig shape should be [none,nx,ny,1,n_labels]
            OUT = Reshape([X.type_spec.shape[1],X.type_spec.shape[2],n_labels],name='squeeze')(OUT)

        model = Model([IN,], [OUT,])
        
    return model

