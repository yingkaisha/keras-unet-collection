
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, patch_merging, patch_expanding

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads, 
                                 window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate, 
                                 name='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021. 
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num_begin: number of channels in the first downsampling block; 
                          it is also the number of embedded dimensions.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches, 
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.

    Output
    ----------
        output tensor.
        
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, stack_num=stack_num_down, 
                               embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=num_mlp, 
                               shift_window=shift_window, name='{}_swin_down0'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_down, 
                                   embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=num_mlp, 
                                   shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1))
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                            embed_dim=embed_dim, upsample_rate=2, return_vector=True, name='{}_swin_up{}'.format(name, i))(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_up, 
                           embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                           num_heads=num_heads[i], window_size=window_size[i], num_mlp=num_mlp, 
                           shift_window=shift_window, name='{}_swin_up{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                        embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
    
    return X


def swin_unet_2d(input_size, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, output_activation='Softmax', shift_window=True, name='swin_unet'):
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021. 
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_begin: number of channels in the first downsampling block; 
                          it is also the number of embedded dimensions.
        n_labels: number of output labels.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches, 
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.
        
    Output
    ----------
        model: a keras model.
    
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    '''
    IN = Input(input_size)
    
    # base    
    X = swin_unet_2d_base(IN, filter_num_begin=filter_num_begin, depth=depth, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                          patch_size=patch_size, num_heads=num_heads, window_size=window_size, num_mlp=num_mlp, shift_window=True, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model
