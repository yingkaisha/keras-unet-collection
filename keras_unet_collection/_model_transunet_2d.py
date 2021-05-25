
from __future__ import absolute_import

from keras_unet_collection.layer_utils import *
from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Embedding

class ViT_patch_gen(Layer):
    '''
    
    '''
    def __init__(self, patch_size):
        super(ViT_patch_gen, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1], padding='VALID',)
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return patches
    
class ViT_embedding(Layer):
    '''
    
    '''
    def __init__(self, num_patches, proj_dim):
        super(ViT_embedding, self).__init__()
        self.num_patches = num_patches
        self.proj = Dense(proj_dim)
        self.pos_embed = Embedding(input_dim=num_patches, output_dim=proj_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.proj(patch) + self.pos_embed(pos)
        return encoded
    
def ViT_MLP(X, filter_num, activation, name='MLP'):
    '''
    
    '''
    activation_func = eval(activation)
    
    for f in filter_num:
        X = Dense(f, name='{}_dense')(X)
        X = activation_func(name='{}_activation')(X)
        
    return X
    
def ViT_block(V, num_heads, key_dim, filter_num_MLP, name='ViT'):
    '''
    
    '''
    # Multi-head self-attention
    V_atten = V # <--- skip
    V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V_atten)
    V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, 
                                 name='{}_atten'.format(name))(V_atten, V_atten)
    # Skip connection
    V_add = add([V_atten, V], name='{}_skip_1'.format(name)) # <--- skip
    
    # MLP
    V_MLP = V_add # <--- skip
    V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_MLP)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, name='{}_mlp'.format(name))
    # Skip connection
    V_out = add([V_MLP, V_add], name='{}_skip_2'.format(name)) # <--- skip
    
    return V_out


def transunet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                      proj_dim=768, num_mlp = 3072, num_heads=12, num_transformer=12,
                      activation='ReLU', batch_norm=False, pool=True, unpool=True, name='transunet'):
    '''
    The base of transUNET.
    
    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
    
    Input
    ----------

        
    Output
    ----------

    
    '''
    activation_func = eval(activation)
    
    X_skip = []
    depth_ = len(filter_num)
    
    # ----- internal parameters ----- #
    
    # patch size (fixed to 1-by-1)
    patch_size = 1
    
    # input tensor size
    input_size = input_tensor.shape[1]
    
    # encoded feature map size
    encode_size = input_size // 2**(depth_-1)
    
    # number of size-1 patches
    num_patches = encode_size ** 2 
    
    # dimension of the attention key (= dimension of embedings)
    key_dim = proj_dim
    
    # number of MLP nodes
    filter_num_MLP = [num_mlp, proj_dim]
    
    # ----- UNet-like downsampling ----- #
    
    X = input_tensor
    
    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                   batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)
    
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                      batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
        X_skip.append(X)
        
    # discard the last tensor (will be replaced by the ViT output)
    X_skip = X_skip[:-1] 
    
    # ----- ViT block after UNet-like encoding ----- #
    
    # 1-by-1 linear transformation before entering ViT blocks
    X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_before'.format(name))(X)
    
    # feature map to patches
    X = ViT_patch_gen(patch_size)(X_skip[-1])
    
    # patches to embeddings
    X = ViT_embedding(num_patches, proj_dim)(X)
    
    # stacked ViTs 
    for i in range(num_transformer):
        X = ViT_block(X, num_heads, key_dim, filter_num_MLP, name='{}_ViT_{}'.format(name, i))
        
    # reshape patches to feature maps
    X = tf.reshape(X, (-1, encode_size, encode_size, proj_dim))
    
    # 1-by-1 linear transformation to adjust the number of channels
    X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_after'.format(name))(X)
    X_skip.append(X)
    
    # ----- UNet-like upsampling ----- #
    
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
        
    return X

def transunet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 proj_dim=768, num_mlp = 3072, num_heads=12, num_transformer=12,
                 activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, name='transunet'):
    '''
    transUNET
    
    ----------
    Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
    Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
    
    Input
    ----------

        
    Output
    ----------

    
    '''
    activation_func = eval(activation)
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
        
    IN = Input(input_size)
    
    # base    
    X = transunet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                          proj_dim=proj_dim, num_mlp=num_mlp, num_heads=num_heads, num_transformer=num_transformer,
                          activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model