# keras-unet-collection

[![PyPI version](https://badge.fury.io/py/keras-unet-collection.svg)](https://badge.fury.io/py/keras-unet-collection)
[![PyPI license](https://img.shields.io/pypi/l/keras-unet-collection.svg)](https://pypi.org/project/keras-unet-collection/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yingkaisha/keras-unet-collection/graphs/commit-activity)

The `tensorflow.keras` implementation of U-net, V-net, U-net++, UNET 3+, Attention U-net, R2U-net, ResUnet-a, U^2-Net, TransUNET, and Swin-UNET with optional ImageNet-trained backbones.

----------

`keras_unet_collection.models` contains functions that configure keras models with hyper-parameter options. 

* Pre-trained ImageNet backbones are supported for U-net, U-net++, UNET 3+, Attention U-net, and TransUNET.
* Deep supervision is supported for U-net++, UNET 3+, and U^2-Net.
* See the [User guide](https://github.com/yingkaisha/keras-unet-collection/blob/main/examples/user_guide_models.ipynb) for other options and use cases.

| `keras_unet_collection.models` | Name | Reference |
|:---------------|:----------------|:----------------|
| `unet_2d`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `vnet_2d`      | V-net (modified for 2-d inputs) | [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) |
| `unet_plus_2d` | U-net++         | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `r2_unet_2d`   | R2U-Net         | [Alom et al. (2018)](https://arxiv.org/abs/1802.06955) |
| `att_unet_2d`  | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
| `resunet_a_2d` | ResUnet-a       | [Diakogiannis et al. (2020)](https://doi.org/10.1016/j.isprsjprs.2020.01.013) |
| `u2net_2d`     | U^2-Net         | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
| `unet_3plus_2d` | UNET 3+        | [Huang et al. (2020)](https://arxiv.org/abs/2004.08790) |
| `transunet_2d` | TransUNET       | [Chen et al. (2021)](https://arxiv.org/abs/2102.04306) |
| `swin_unet_2d` | Swin-UNET       | [Hu et al. (2021)](https://arxiv.org/abs/2105.05537) |

**Note**: the two Transformer models are incompatible with `Numpy 1.20`; `NumPy 1.19.5` is recommended.

----------

` keras_unet_collection.base` contains functions that build the base architecture (i.e., without model heads) of Unet variants for model customization and debugging.

| ` keras_unet_collection.base` | Notes |
|:-----------------------------------|:------|
| `unet_2d_base`, `vnet_2d_base`, `unet_plus_2d_base`, `unet_3plus_2d_base`, `att_unet_2d_base`, `r2_unet_2d_base`, `resunet_a_2d_base`, `u2net_2d_base`, `transunet_2d_base`, `swin_unet_2d_base` | Functions that accept an input tensor and hyper-parameters of the corresponded model, and produce output tensors of the base architecture. |

----------

`keras_unet_collection.activations` and `keras_unet_collection.losses` provide additional activation layers and loss functions.

| `keras_unet_collection.activations` | Name | Reference |
|:--------|:----------------|:----------------|
| `GELU`  | Gaussian Error Linear Units (GELU)   | [Hendrycks et al. (2016)](https://arxiv.org/abs/1606.08415) |
| `Snake` | Snake activation                     | [Liu et al. (2020)](https://arxiv.org/abs/2006.08195) |

| `keras_unet_collection.losses` | Name | Reference |
|:----------------|:----------------|:----------------|
| `dice`          | Dice loss                      | [Sudre et al. (2017)](https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28) |
| `tversky`       | Tversky loss                   | [Hashemi et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8573779) |
| `focal_tversky` | Focal Tversky loss             | [Abraham et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8759329) |
| `ms_ssim`       | Multi-scale Structural Similarity Index loss | [Wang et al. (2003)](https://ieeexplore.ieee.org/abstract/document/1292216) |
| `iou_seg`       | Intersection over Union (IoU) loss for segmentation | [Rahman and Wang (2016)](https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22) |
| `iou_box`       | (Generalized) IoU loss for object detection | [Rezatofighi et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html) |
| `triplet_1d`    | Semi-hard triplet loss (experimental) | |
| `crps2d_tf`     | CRPS loss (experimental)       | |

# Installation and usage

```pip install keras-unet-collection```

```python
from keras_unet_collection import models
# e.g. models.unet_2d(...)
```
* **Note**: Currently supported backbone models are: `VGG[16,19]`, `ResNet[50,101,152]`, `ResNet[50,101,152]V2`, `DenseNet[121,169,201]`, and `EfficientNetB[0-7]`. See [Keras Applications](https://keras.io/api/applications/) for details. 

* **Note**: Neural networks produced by this package may contain customized layers that are not part of the Tensorflow. It is reommended to save and load model weights.

Keras models created by this package may contain customized layers that are not 

* Jupyter notebooks are provided as [examples](https://github.com/yingkaisha/keras-unet-collection/tree/main/examples):

  * Attention U-net with VGG16 backbone [[link]](https://github.com/yingkaisha/keras-unet-collection/blob/main/examples/human-seg_atten-unet-backbone_coco.ipynb).
  
  * UNET 3+ with deep supervision, classification-guided module, and hybrid loss [[link]](https://github.com/yingkaisha/keras-unet-collection/blob/main/examples/segmentation_unet-three-plus_oxford-iiit.ipynb).

  * Vision-Transformer-based examples are in progress, and available at [**keras-vision-transformer**](https://github.com/yingkaisha/keras-vision-transformer)

* [Changelog](https://github.com/yingkaisha/keras-unet-collection/blob/main/CHANGELOG.md)

# Dependencies

* TensorFlow 2.5.0, Keras 2.5.0, Numpy 1.19.5.

* (Optional for examples) Pillow, matplotlib, etc.

# Overview

U-net is a convolutional neural network with encoder-decoder architecture and skip-connections, loosely defined under the concept of "fully convolutional networks." U-net was originally proposed for the semantic segmentation of medical images and is modified for solving a wider range of gridded learning problems.

U-net and many of its variants take three or four-dimensional tensors as inputs and produce outputs of the same shape. One technical highlight of these models is the skip-connections from downsampling to upsampling layers, which benefit the reconstruction of high-resolution, gridded outputs.

# Contact

Yingkai (Kyle) Sha <<yingkai@eoas.ubc.ca>> <<yingkaisha@gmail.com>>

# License

[MIT License](https://github.com/yingkaisha/keras-unet/blob/main/LICENSE)
