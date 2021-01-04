# keras-unet-collection

This repository contains `tensorflow.keras` implementations of U-net, U-net++, R2U-net, Attention U-net, ResUnet-a:

| `keras_unet_collection.models`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Reference |
|:-------------------------------|:----------|
| U-net/Unet      | Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. |
| U-net++/Unet++  | Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 3-11). Springer, Cham. |
| R2U-Net         | Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955. |
| Attention U-net | Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B. and Glocker, B., 2018. Attention 
| ResUnet-a       | Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114. |

These models are implemented with user friendly key words, including optional network depth, hidden layer activations and batch normalization. Examples refers to the [user guide](https://github.com/yingkaisha/keras-unet-collection/blob/main/user_guid.ipynb).

Additional activation layers and loss functions are also provided:

| ` keras_unet_collection.activations` | Reference |
|:------------------------------------|:----------|
| Gaussian Error Linear Units (GELU)  | Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415. |
| Snake activation                    | Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions and how to fix it. arXiv preprint arXiv:2006.08195. |

| `keras_unet_collection.losses`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8239; | Reference |
|:-------------------------------|:----------|
| Tversky loss                   | Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018. Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks. arXiv preprint arXiv:1803.11078. |
| Focal Tversky loss             | Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019) (pp. 683-687). IEEE. |
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
