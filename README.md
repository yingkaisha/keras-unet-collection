# keras-unet-collection

[![PyPI version](https://badge.fury.io/py/keras-unet-collection.svg)](https://badge.fury.io/py/keras-unet-collection)
[![PyPI license](https://img.shields.io/pypi/l/keras-unet-collection.svg)](https://pypi.org/project/keras-unet-collection/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yingkaisha/keras-unet-collection/graphs/commit-activity)

This repository contains `tensorflow.keras` implementations of U-net, U-net++, R2U-net, Attention U-net, ResUnet-a, U^2-Net:

| `keras_unet_collection.models`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Name | Reference |
|:---------------|:----------------|:----------------|
| `unet_2d`      | U-net/Unet      | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `unet_plus_2d` | U-net++/Unet++  | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `r2_unet_2d`   | R2U-Net         | [Alom et al. (2018)](https://arxiv.org/abs/1802.06955) |
| `att_unet_2d`  | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
| `resunet_a_2d` | ResUnet-a       | [Diakogiannis et al. (2020)](https://doi.org/10.1016/j.isprsjprs.2020.01.013) |
| `u2net_2d`     | U^-Net          | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |

These models are implemented with user-friendly hyper-parameter options and keywords, including network depth, hidden layer activations and batch normalization. [User guide](https://github.com/yingkaisha/keras-unet-collection/blob/main/user_guid.ipynb) provided several examples.

Additional activation layers and loss functions are also provided:

| ` keras_unet_collection.activations` | Name | Reference |
|:--------|:----------------|:----------------|
| `GELU`  | Gaussian Error Linear Units (GELU)   | [Hendrycks et al. (2016)](https://arxiv.org/abs/1606.08415) |
| `Snake` | Snake activation                     | [Liu et al. (2020)](https://arxiv.org/abs/2006.08195) |

| `keras_unet_collection.losses`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Name | Reference |
|:----------------|:----------------|:----------------|
| `tversky`       | Tversky loss                   | [Hashemi et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8573779) |
| `focal_tversky` | Focal Tversky loss             | [Abraham et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8759329) |
| `crps2d_tf`     | CRPS loss (experimental)       | |

# Dependencies

* TensorFlow 2.3.0

* Keras 2.4.0

* Numpy 1.18.2

# Installation and usage

```pip install keras-unet-collection```

```python
from keras_unet_collection import models
# e.g. models.unet_2d(...)
```

* Jupyter notebooks are provided as [user guides](https://github.com/yingkaisha/keras-unet-collection/blob/main/user_guid.ipynb).

> Note: Because of the flexible hyper-parameter options, neural networks produced by this package may not be compatible with other pre-trained models of the same name/kind. Training from scratch is recommended.

* [Changelog](https://github.com/yingkaisha/keras-unet-collection/blob/main/CHANGELOG.md)

# Overview

U-net is a convolutional neural network with encoder-decoder architecture and skip-connections, loosely defined under the concept of "fully convolutional networks." U-net was originally proposed for the semantic segmentation of medical images and is modified for solving a wider range of gridded learning problems.

U-net and many of its variants take three or four-dimensional tensors as inputs and produce outputs of the same shape. One technical highlight of these models is the skip-connections from downsampling to upsampling layers, which benefit the reconstruction of high-resolution, gridded outputs.

# Contact

Yingkai (Kyle) Sha <yingkai@eoas.ubc.ca>

# License

[MIT License](https://github.com/yingkaisha/keras-unet/blob/main/LICENSE)
