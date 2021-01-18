
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model


def UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):
    '''
    Decoder block of Attention U-net
    
    UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0')
    
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

def att_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add',
                     batch_norm=False, pool=True, unpool=True, name='att-unet'):
    '''
    The base of Attention U-net
    
    att_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add',
                     batch_norm=False, pool=True, unpool=True, name='att-unet')
                
    ----------
    Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B. 
    and Glocker, B., 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num: an iterable that defines the number of filters for each \
                      down- and upsampling level. E.g., [64, 128, 256, 512]
                      the depth is expected as `len(filter_num)`
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU        
        atten_activation: a nonlinear atteNtion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is ReLU
        attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: the output tensor of the base.
    
    '''
    
    activation_func = eval(activation)

    X = input_tensor
    X_skip = []
    
    # downsampling blocks
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)
    
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                      batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
        X_skip.append(X)
        
    # upsampling blocks
    X_skip = X_skip[:-1][::-1]
    for i, f in enumerate(filter_num[:-1][::-1]):
        X = UNET_att_right(X, X_skip[i], f, att_channel=f//2, stack_num=stack_num_up,
                           activation=activation, atten_activation=atten_activation, attention=attention,
                           unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i+1))
    
    return X

def att_unet_2d(input_size, filter_num, n_labels,
                stack_num_down=2, stack_num_up=2,
                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
                batch_norm=False, pool=True, unpool=True, name='att-unet'):
    '''
    Attention U-net
    
    att_unet_2d(input_size, filter_num, n_labels,
                stack_num_down=2, stack_num_up=2,
                activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
                batch_norm=False, pool=True, unpool=True, name='att-unet')
                
    ----------
    Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B. 
    and Glocker, B., 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.
    
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
        atten_activation: a nonlinear atteNtion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is ReLU
        attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        batch_norm: True for batch normalization.
        pool: True for maxpooling, False for strided convolutional layers.
        unpool: True for unpooling (i.e., reflective padding), False for transpose convolutional layers.                 
        name: prefix of the created keras layers.
        
    Output
    ----------
        model: a keras model 
    
    '''
    
    # one of the ReLU, LeakyReLU, PReLU, ELU
    activation_func = eval(activation)
    
    IN = Input(input_size)
    
    # base
    X = att_unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                         activation=activation, atten_activation=atten_activation, attention=attention,
                         batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    
    return model
