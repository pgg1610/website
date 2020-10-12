---
description: Example of how to implement a convolutional neural network in PyTorch
fact: 
featured: true
image: /img/pytorch.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/PyTorch_tutorial/cnn_pytorch_tutorial.ipynb
tags:
- Python
- PyTorch

title: PyTorch CNN example
---
Convolutional neural network is used to train on the CIFAR-10 dataset using PyTorch. 

### What does it consists of?
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

### What's a Convolutional Neural Network (CNN)?

Same as a neural network but are optimized for image analysis. Before training the weights and biases in the full-connected layer the training data is 'screened and filtered' to tease out relevant features of each image by passing each image through a prescribed filter and 'convolutions'. Think of it like passing a colored lens or fancy ink to selectively look at edges, contrasts, shapes in the image.

Finally that a projection of that image is made by 'pooling' which is a way of down-sampling the resulting convolution as a new data-point.

Good reading links:
* http://cs231n.github.io/convolutional-networks/
* https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/