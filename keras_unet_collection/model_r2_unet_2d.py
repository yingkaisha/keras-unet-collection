
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model

def r2_unet_2d(input_size, filter_num, n_labels, 
               stack_num_down=2, stack_num_up=2, recur_num=2,
               activation='ReLU', output_activation='Softmax', 
               batch_norm=False, pool=True, unpool=True, name='res_unet'):
    
    '''
    Recurrent Residual (R2) U-Net
    
    r2_unet_2d(input_size, filter_num, n_labels, 
               stack_num_down=2, stack_num_up=2, recur_num=2,
               activation='ReLU', output_activation='Softmax', 
               batch_norm=False, pool=True, unpool=True, name='r2_unet')
    
    ----------
    Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network 
    based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.
    
    Input
    ----------
        input_size: a tuple that defines the shape of input, e.g., (None, None, 3)
        filter_num: an iterable that defines number of filters for each \
                      down- and upsampling level. E.g., [64, 128, 256, 512]
                      the depth is expected as `len(filter_num)`
        n_labels: number of output labels.
        
        stack_num_down: number of stacked recurrent convolutional layers per downsampling level/block.
        stack_num_down: number of stacked recurrent convolutional layers per upsampling level/block.
        recur_num: number of recurrent iterations.
        
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'.
                           Default option is Softmax
                           if None is received, then linear activation is applied.
                           
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: a keras model 
    
    '''
    
    activation_func = eval(activation)

    IN = Input(input_size, name='{}_input'.format(name))
    X = IN
    X_skip = []
    
    # downsampling blocks
    X = RR_CONV(X, filter_num[0], stack_num=stack_num_down, recur_num=recur_num, 
                      activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)
    
    for i, f in enumerate(filter_num[1:]):
        X = UNET_RR_left(X, f, kernel_size=3, stack_num=stack_num_down, recur_num=recur_num, 
                          activation='ReLU', pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
        X_skip.append(X)
    
    # upsampling blocks
    X_skip = X_skip[:-1][::-1]
    for i, f in enumerate(filter_num[:-1][::-1]):
        X = UNET_RR_right(X, [X_skip[i],], f, stack_num=stack_num_up, recur_num=recur_num, 
                           activation='ReLU', unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i+1))

    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    
    return model 