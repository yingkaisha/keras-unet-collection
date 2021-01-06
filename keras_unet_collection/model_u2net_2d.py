
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model

def u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
             filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
             batch_norm=False, deep_supervision=False, name='u2net'):
    
    '''
    U^2-Net
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        input_size: a tuple that defines the shape of input, e.g., (None, None, 3)
        
        filter_num_down: an iterable that defines number of RSU output filters for each 
                         downsampling level. E.g., [64, 128, 256, 512]
                         the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`
                         
        filter_num_down: an iterable that defines number of RSU output filters for each 
                         upsampling level. E.g., [64, 128, 256, 512].
                         
        filter_mid_num_down: an iterable that defines number of RSU intermediate filters for each 
                             downsampling level. E.g., [16, 32, 64, 128]
                             *RSU intermediate and output filters must paired, i.e., list with the same length
                             *RSU intermediate filters numbers are expected to be smaller than output filters numbers
                             
        filter_mid_num_up: an iterable that defines number of RSU intermediate filters for each 
                             upsampling level. E.g., [16, 32, 64, 128]
                             *RSU intermediate and output filters must paired, i.e., list with the same length
                             *RSU intermediate filters numbers are expected to be smaller than output filters numbers
              
        filter_4f_num: an iterable that defines number of RSU-4F output filters for each 
                       downsampling and bottom level. E.g., [512, 512]
                       the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`       
        
        filter_4f_mid_num: an iterable that defines number of RSU-4F intermediate filters for each 
                           downsampling and bottom level. E.g., [256, 256]
                           *RSU-4F intermediate and output filters must paired, i.e., list with the same length
                           *RSU-4F intermediate filters numbers are expected to be smaller than output filters numbers
                         
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'. 
                           Default option is Softmax
                           if None is received, then linear activation is applied.
                           
        batch_norm: True for batch normalization.
        deep_supervision: True for a model that supports deep supervision. Details see Qin et al. (2020).
        name: prefix of the created keras layers.    
    
    * "auto" mode will produce a slightly larger network, not compateble with Qin et al. (2020). 
    * downsampling is achieved through maxpooling (Qin et al., 2020).
    * upsampling is achieved through bilinear interpolation (Qin et al., 2020).
    
    '''
    
    verbose = False
    
    if filter_num_up == 'auto':
        verbose = True
        filter_num_up = filter_num_down
        
    if filter_mid_num_down == 'auto':
        verbose = True
        filter_mid_num_down = [num//4 for num in filter_num_down]
        
    if filter_mid_num_up == 'auto':
        verbose = True
        filter_mid_num_up = filter_mid_num_down
        
    if filter_4f_num == 'auto':
        verbose = True
        filter_4f_num = [filter_num_down[-1], filter_num_down[-1]]
        
    if filter_4f_mid_num == 'auto':
        verbose = True
        filter_4f_mid_num = [num//2 for num in filter_4f_num]
        
    if verbose:
        print('"auto" mode is applied with the following details:\n----------')
        print('\tNumber of RSU outer channels within downsampling blocks: filter_num_down = {}'.format(filter_num_down))
        print('\tNumber of RSU intermediate channels within downsampling blocks: filter_mid_num_down = {}'.format(filter_mid_num_down))
        print('\tNumber of RSU outer channels within upsampling blocks: filter_num_up = {}'.format(filter_num_up))
        print('\tNumber of RSU intermediate channels within upsampling blocks: filter_mid_num_up = {}'.format(filter_mid_num_up))        
        print('\tNumber of RSU-4F outer channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_num))
        print('\tNumber of RSU-4F intermediate channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_mid_num))
        print('----------\nExplicitly specifying keywords listed above if their "auto" settings do not satisfy your needs')
        
    print("----------\nThe depth of u2net_2d = len(filter_num_down) + len(filter_4f_num) = {}".format(len(filter_num_down)+len(filter_4f_num)))
    
    X_skip = []; X_out = []; OUT_stack = []
    depth_backup = []
    
    depth_ = len(filter_num_down)
    
    IN = Input(shape=input_size) 
    X = IN
    
    X = RSU(X, filter_mid_num_down[0], filter_num_down[0], depth=depth_+1, 
            activation=activation, batch_norm=batch_norm, name='{}_in'.format(name))
    X_skip.append(X)
    depth_backup.append(depth_+1)
    
    for i, f in enumerate(filter_num_down[1:]):
        X = MaxPooling2D(pool_size=(2, 2), name='{}_maxpool_{}'.format(name, i))(X)
        X = RSU(X, filter_mid_num_down[i+1], f, depth=depth_-i, 
                activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i))
        depth_backup.append(depth_-i)
        X_skip.append(X)

    for i, f in enumerate(filter_4f_num):
        X = MaxPooling2D(pool_size=(2, 2), name='{}_maxpool_4f_{}'.format(name, i))(X)
        X = RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_down_4f_{}'.format(name, i))
        X_skip.append(X)
        
    X_out.append(X)
    
    # ---------- #
    X_skip = X_skip[:-1][::-1]
    depth_backup = depth_backup[::-1]
    
    filter_num_up = filter_num_up[::-1]
    filter_mid_num_up = filter_mid_num_up[::-1]
    
    filter_4f_num = filter_4f_num[:-1][::-1]
    filter_4f_mid_num = filter_4f_mid_num[:-1][::-1]
    
    tensor_count = 0
    for i, f in enumerate(filter_4f_num):
        X = UpSampling2D((2, 2), interpolation='bilinear', name='{}_unpool_4f_{}'.format(name, i))(X)
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_4f_{}'.format(name, i))
        X = RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_up_4f_{}'.format(name, i))
        X_out.append(X)
        tensor_count += 1
    
    for i, f in enumerate(filter_num_up):
        X = UpSampling2D((2, 2), interpolation='bilinear', name='{}_unpool_{}'.format(name, i))(X)
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = RSU(X, filter_mid_num_up[i], f, depth=depth_backup[i], 
                activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))
        X_out.append(X)
        tensor_count += 1
    
    # ---------- #
    X_out = X_out[::-1]
    L_out = len(X_out)
    
    D = X_out[0]
    D = CONV_output(D, n_labels, kernel_size=3, activation=output_activation, 
                    name='{}_output_sup0'.format(name))
    OUT_stack.append(D)
    
    for i in range(L_out-1):
        pool_size = 2**(i+1)
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i))(X_out[i+1])
        D = UpSampling2D((pool_size, pool_size), interpolation='bilinear', name='{}_output_sup{}'.format(name, i+1))(X)
        
        if output_activation:
            if output_activation == 'Sigmoid':
                D = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i+1))(D)
            else:
                activation_func = eval(output_activation)
                D = activation_func(name='{}_output_sup{}_activation'.format(name, i+1))(D)
        OUT_stack.append(D)
        
    D = concatenate(OUT_stack, axis=-1, name='{}_output_concat'.format(name))
    D = CONV_output(D, n_labels, kernel_size=1, activation=output_activation, 
                    name='{}_output_final'.format(name))
    
    if deep_supervision:
        OUT_stack.append(D)
        print('----------\ndeep_supervision = True\nnames of (ordered) output tensors are listed as follows:')
        
        if output_activation == None:
            for i in range(L_out):
                print('\t{}_output_sup{}'.format(name, i))
            print('\t{}_output_final'.format(name))
        
        else:        
            for i in range(L_out):
                print('\t{}_output_sup{}_activation'.format(name, i))
            print('\t{}_output_final_activation'.format(name))
            
        model = Model([IN], OUT_stack)
        
    else:
        model = Model([IN], [D])
        
    return model

