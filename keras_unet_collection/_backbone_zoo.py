
from __future__ import absolute_import

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

from keras_unet_collection.utils import freeze_model

X_skip_cadidates = {
    'VGG16': ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19': ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation')
}

def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`
    
    Input
    ----------
        backbone_name: 
        weights:
        input_tensor:
        depth:
    '''
    
    cadidate = X_skip_cadidates[backbone_name]
    
    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max or depth < 2:
        depth = depth_max
    # ----- #
    
    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None,)
    
    X_skip = []
    
    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)
        
    model = Model(inputs=[input_tensor,], outputs=X_skip, name='{}_backbone'.format(backbone_name))
    
    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)
    
    return model
