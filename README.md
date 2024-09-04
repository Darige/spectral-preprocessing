## Spectral-preprocessing and augmentation methods for deep learning
Code for my UCL dissertation


Note: The Jupyter Notebook ('ROF-DENOISING.ipynb') gives an introduction to the total variation denoising problem and the spectral framework derived from it.


All the packages required to run this repositry is available in:

```
requirements.txt
```

All the helper functions for our experiments can be found in:

```
pock.py
```
## Image classification

We perform image classification on CIFAR-10, flowers102 and Imagenette.

# CIFAR-10 code

Base model
```
train_pt.py
```
Data preprocessed version
```
train_pt_aug.py
```
Test script
```
test_pt.py
```

# flowers102 code

Base model
```
train_flowers102.py
```
Data preprocessed version
```
train_aug_flowers102.py
```
Test script
```
test_flowers102.py
```


# Imagenette Code

Base model
```
train_imagenette.py
```
Data preprocessed version
```
train_aug_imagenette.py
```
Test script
```
test_imagenette.py
```



## Image Generation

We also study the effect of spectral augmentation methods on image generation. 
To do so we train a generative adversarial network (GAN) on celeb-a dataset.

The code for training can be found here:
```
gan.py
```
The code for generating the images can be found here:
```
gan_test.py
```








# Abstract:

This dissertation explores two uses of the total variation spectral framework in image classification and image generation. 

Total variation denoising is a noise removing method and is based on the principle that noisy images often have high total variation among them, thus by reducing the total variation of the image, we will end up with something similar to the original image. 

A spectral framework in the context of our dissertation can be thought of as a method which is used to decompose an image into multiple components while providing information about each of the specific components. Due to our spectral framework being based on total variation, each of the components is able to extract specific features.

Given this information, in the case of image classification, we are able to identify and amplify the specific spectral components, we can then reconstruct the image with the attenuated components into a processed image. We then feed both the processed image together with the original image into an image classification model for training. We seek to explore if this data preprocessing step could improve the performance of the image classification model compared to the native method of training these models, which involved feeding just the original image into the model for training.

In the case of the image generation model, we train each of the image generation models on each unique spectral component. After training the image generation models, each generator is seeded to compute the spectral components of the same set of images. Using these generated spectral components we can reconstruct an image. We wish to compare if this reconstructed image looks more realistic compared to an image generated from a generator trained from just the original dataset.  




