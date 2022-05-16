from keras_unet_collection import models
from numpy.testing import assert_almost_equal
import numpy as np 
import tensorflow as tf


#run simple model to get weights and check they are initialized right 
model = models.unet_2d((32, 32, 1), [4], n_labels=1,kernel_size=3,
                      stack_num_down=1, stack_num_up=1,
                      activation="ReLU", output_activation='Sigmoid',weights=None,
                      batch_norm=False, pool='max', unpool='nearest', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                       name='unet')

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.binary_crossentropy,)


def test_initialization_seed():
    #get initalized weights
    weights = model.get_weights()

    #get CNN kernel weights 
    l0_k0 = weights[0][:,:,0,0]
    l0_k1 = weights[0][:,:,0,1]

    #check to expected weights 
    l0_k0_expected = np.array([[ 0.33029234, -0.01755953, -0.11960667],
       [ 0.23655796,  0.00443587,  0.23982102],
       [ 0.05338666, -0.34987593,  0.11066625]], dtype=np.float32)
    l0_k1_expected = np.array([[ 0.12956029,  0.0956797 ,  0.16310084],
       [ 0.06967962, -0.10108131, -0.062379  ],
       [-0.19261172,  0.33730918,  0.2680394 ]], dtype=np.float32)

    #get difference 
    l0_k0_diff = np.sum(l0_k0 - l0_k0_expected)
    l0_k1_diff = np.sum(l0_k1 - l0_k1_expected)

    #check to make sure they match (to 8 decimals)
    assert_almost_equal(l0_k0_diff, 0,decimal=8)
    assert_almost_equal(l0_k1_diff, 0,decimal=8)

    #get bias weights 
    l0_b0 = weights[1][0]
    l1_b0 = weights[3][0]

    #check to make sure they match (to 8 decimals)
    assert_almost_equal(l0_b0, 0,decimal=8)
    assert_almost_equal(l1_b0, 0,decimal=8)

