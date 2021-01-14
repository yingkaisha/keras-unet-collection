
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model

def unet_3plus_2d_backbone(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                           stack_num_down=2, stack_num_up=1, activation='ReLU', 
                           batch_norm=False, pool=True, unpool=True, name='unet3plus'):
    '''
    The backbone of U-net+++
    
    unet_3plus_2d_backbone(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                           stack_num_down=2, stack_num_up=1, activation='ReLU', 
                           batch_norm=False, pool=True, unpool=True, name='unet3plus')
                  
    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.
    
    Input
    ----------
        input_tensor: the input tensor of the backbone, e.g., keras.layers.Inpyt((None, None, 3))
        
        filter_num_down: an iterable that defines the number of RSU output filters for each 
                         downsampling level. E.g., [64, 128, 256, 512, 1024]
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: an iterable that defines the number of convolution filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           E.g., [64, 64, 64, 64]
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.          
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU                
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling with bilinear interpolation, False for transpose convolutional layers.  
        name: prefix of the created keras layers.    

    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.
    
    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with first tensor has the smallest size. 
    
    '''
    
    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    X = input_tensor
    
    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down, 
                   activation=activation, batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_encoder.append(X)
    
    # downsampling levels
    for i, f in enumerate(filter_num_down[1:]):
        # ----- #
        if pool:
            X = MaxPooling2D(pool_size=(2, 2), name='{}_maxpool_{}'.format(name, i+1))(X)
        
        else:
            X = stride_conv(X, f, pool_size=2, activation=activation, 
                            batch_norm=batch_norm, name='{}_stridconv_{}'.format(name, i+1))
        # ----- #
        
        X = CONV_stack(X, f, kernel_size=3, stack_num=stack_num_down, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))
    
        X_encoder.append(X)
    
    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])
    
    # upsampling levels
    X_encoder = X_encoder[::-1]

    # loop over upsampling levels
    for i, f in enumerate(filter_num_skip):
    
        # collecting tensors for layer fusion
        X_fscale = []
    
        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_):
        
            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev-i-1 # -1 for python indexing
        
            # one scale deeper input is obtained from the nearest **decoder** output
            if pool_scale == -1:
            
                if unpool:
                    X = UpSampling2D(size=(2, 2), interpolation='bilinear', 
                                     name='{}_unpool_{}_de{}'.format(name, i, i))(X_decoder[i])
                else:
                    X = Conv2DTranspose(f, kernel_size=3, strides=(2, 2), padding='same', 
                                        name='{}_transconv_{}_de{}'.format(name, i, i))(X_decoder[i])
            
            # other inputs are obtained from **encoder** outputs
            else:
                # deeper tensors are upsampled
                if pool_scale < 0:
                    pool_size = 2**(-1*pool_scale)
                    
                    if unpool:
                        X = UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear', 
                                         name='{}_unpool_{}_en{}'.format(name, i, lev))(X_encoder[lev])
                    else:
                        X = Conv2DTranspose(f, kernel_size=3, strides=(pool_size, pool_size), padding='same', 
                                            name='{}_transconv_{}_en{}'.format(name, i, lev))(X_encoder[lev])
                # unet skip connection (identity mapping)    
                elif pool_scale == 0:
                    
                    X = X_encoder[lev]
                    
                # shallower tensors are downsampled
                else:
                    pool_size = 2**(pool_scale)

                    if pool:
                        X = MaxPooling2D(pool_size=(pool_size, pool_size), 
                                         name='{}_maxpool_{}_en{}'.format(name, i, lev))(X_encoder[lev])
                    else:
                        X = stride_conv(X_encoder[lev], f, pool_size=pool_size, activation=activation, 
                            batch_norm=batch_norm, name='{}_stridconv_{}_en{}'.format(name, i, lev))
                        
            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=3, stack_num=1, 
                           activation=activation, batch_norm=batch_norm, name='{}_down_from{}_to{}'.format(name, i, lev))
        
            X_fscale.append(X)  

        # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True
        
        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, filter_num_aggregate, kernel_size=3, stack_num=stack_num_up, 
                       activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        X_decoder.append(X)
        
    # return decoder outputs
    return X_decoder

def unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='unet3plus'):
    
    '''
    U-net+++ (U-net three plus; UNet 3+)
    
    unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Softmax',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='unet3plus')
                  
    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.
    
    Input
    ----------
        input_size: a tuple that defines the shape of input, e.g., (None, None, 3)
        
        filter_num_down: an iterable that defines the number of RSU output filters for each 
                         downsampling level. E.g., [64, 128, 256, 512, 1024]
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: an iterable that defines the number of convolution filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           E.g., [64, 64, 64, 64]
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
              
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
                         
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'. 
                           Default option is Softmax
                           if None is received, then linear activation is applied.
                           
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling with bilinear interpolation, False for transpose convolutional layers.
        deep_supervision: True for a model that supports deep supervision. Details see Huang et al. (2020).
        name: prefix of the created keras layers.    
    
    * Automated mode is applied for determining `filter_num_skip`, `filter_num_aggregate`.
    * The default output activation is sigmoid, consistent with Huang et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.
    
    Output
    ----------
        model: a keras model
    
    '''

    depth_ = len(filter_num_down)
    
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

    X_encoder = []
    X_decoder = []


    IN = Input(input_size)

    X_decoder = unet_3plus_2d_backbone(IN, filter_num_down, filter_num_skip, filter_num_aggregate, 
                                       stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation, 
                                       batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    X_decoder = X_decoder[::-1]
    
    if deep_supervision:
        OUT_stack = []
        L_out = len(X_decoder)
        
        print('----------\ndeep_supervision = True\nnames of output tensors are listed as follows (the last one is the final output):')
        
        # conv2d --> upsampling --> output activation.
        # * the bottom level tensor is excluded.
        for i in range(1, L_out-1):
            
            
            pool_size = 2**(i)

            X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i-1))(X_decoder[i])

            if unpool:
                X = UpSampling2D((pool_size, pool_size), interpolation='bilinear', 
                                 name='{}_output_sup{}'.format(name, i-1))(X)
            else:
                X = Conv2DTranspose(n_labels, 3, strides=(pool_size, pool_size), padding='same', 
                                    name='{}_output_sup{}'.format(name, i-1))(X)

            if output_activation:
                print('\t{}_output_sup{}_activation'.format(name, i-1))
                
                if output_activation == 'Sigmoid':
                    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
                else:
                    activation_func = eval(output_activation)
                    X = activation_func(name='{}_output_sup{}_activation'.format(name, i-1))(X)
            else:
                print('\t{}_output_sup{}'.format(name, i-1))
                
            OUT_stack.append(X)

        OUT_stack.append(
            CONV_output(X_decoder[0], n_labels, kernel_size=3, 
                        activation=activation, name='{}_output_final'.format(name)))
        if output_activation:
            print('\t{}_output_final_activation'.format(name))
        else:
            print('\t{}_output_final'.format(name))
            
        model = Model([IN,], OUT_stack)

    else:
        OUT = CONV_output(X_decoder[0], n_labels, kernel_size=3, 
                          activation=activation, name='{}_output_final'.format(name))

        model = Model([IN,], [OUT,])
        
    return model

