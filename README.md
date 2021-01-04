# keras-unet-collection

This repository contains `tensorflow.keras` implementations of U-net, U-net++, R2U-net, Attention U-net, ResUnet-a:

| `keras_unet_collection.models`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Reference |
|:-------------------------------|:----------|
| U-net/Unet      | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| U-net++/Unet++  | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| R2U-Net         | [Alom et al. (2018)](https://arxiv.org/abs/1802.06955) |
| Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
| ResUnet-a       | [Diakogiannis et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0924271620300149?casa_token=NEyVYTeBykMAAAAA:sDSnOwMCM2Q7iucSrKQrefUxr4LFNWHnMSXlSDqaoq8NTbsrqruNUAS6_WxhXnWOOWj0FRoKRA) |

These models are implemented with user friendly key words, including optional network depth, hidden layer activations and batch normalization. Examples refers to the [user guide](https://github.com/yingkaisha/keras-unet-collection/blob/main/user_guid.ipynb).

Additional activation layers and loss functions are also provided:

| ` keras_unet_collection.activations` | Reference |
|:-------------------------------------|:----------|
| Gaussian Error Linear Units (GELU)   | [Hendrycks et al. (2016)](https://arxiv.org/abs/1606.08415) |
| Snake activation                     | [Liu et al. (2020)](https://arxiv.org/abs/2006.08195) |

| `keras_unet_collection.losses`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Reference |
|:-------------------------------|:----------|
| Tversky loss                   | [Hashemi et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8573779) |
| Focal Tversky loss             | [Abraham et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8759329) |
| CRPS loss (experimental)       | |

# Dependencies

* TensorFlow 2.3.0

* Keras 2.4.0

# Installation and usage

```pip install keras-unet-collection```

```python
from keras_unet_collection import models
# e.g. models.unet_2d(...)
```

* Jupyter notebooks are provided as [user guides](https://github.com/yingkaisha/keras-unet-collection/blob/main/user_guid.ipynb).

| Version  | Release date  | Update  |
|:--------:|:-------------:|:-------- |
| 0.0.2    | 2020-12-30    | (1) CRPS loss function.<br />(2) Semi-hard triplet loss function.<br />(3) Fixing user specified names on keras models. |
| 0.0.3    | 2020-01-01    | (1) Bug fix.<br />(2) keyword and documentation fixes for R2U-Net. |

# Overview

U-net is a convolutional neural network with encoder-decoder architecture and skip-connections, loosely defined under the concept of "fully convolutional networks." U-net was originally proposed for the semantic segmentation of medical images and is modified for solving a wider range of gridded learning problems.

U-net and many of its variants take three or four-dimensional tensors as inputs and produce outputs of the same shape. One technical highlight of these models is the skip-connections from downsampling to upsampling layers that benefit the reconstruction of high-resolution, gridded outputs.

# Contact

Yingkai (Kyle) Sha <yingkai@eoas.ubc.ca>

# License

[MIT License](https://github.com/yingkaisha/keras-unet/blob/main/LICENSE)
