# CNN for Image Classification on CIFAR10

## Project Overview
This project aims to build and compare different Convolutional Neural Network (CNN) models for image classification using the CIFAR10 dataset. The project explores custom dataset creation, data preprocessing, model development, training, and evaluation phases to understand and optimize CNNs for real-world image classification tasks.

## Dataset
### Custom Dataset Creation
- **Objective**: Create a custom dataset featuring three distinct categories, each containing at least 100 images.
- **Split**: The dataset is split into 80% training data and 20% testing data.

### CIFAR10
- The CIFAR10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project uses this dataset to benchmark the CNN models developed.

## Preprocessing
Images are preprocessed for the neural network. The typical preprocessing steps involve:
- Resizing images to fit the input shape of the model.
- Normalizing pixel values.
- Data augmentation to prevent overfitting, which includes techniques like rotation, translation, and horizontal flipping.

## Models
### Model 1: Basic CNN
- A simple CNN model designed to understand basic image classification.

### Model 2: CNN with Data Augmentation
- Similar to Model 1 but includes data augmentation layers to improve generalization and avoid overfitting.

### Model 3: GoogleNet (InceptionNet)
- Advanced model using GoogleNet architecture with an additional linear layer to tailor the model to our specific task.
- GoogleNet, also known as Inception v1, is a deep convolutional neural network architecture that was introduced by researchers at Google in 2014. It won the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) by a significant margin. The architecture is notable for its complexity and its efficient use of computation resources.

### Key Features of GoogleNet:

1. **Inception Modules**: The core idea of GoogleNet is the inception module, which allows the network to choose from different types of filters (e.g., 1x1, 3x3, 5x5 convolutions) within the same layer. Each inception module applies these convolutions in parallel and then concatenates their outputs along the channel dimension.

2. **Dimension Reduction**: GoogleNet uses 1x1 convolutions as a method to reduce dimensionality, both in terms of depth (number of channels) and computational complexity. This allows the network to increase its depth and width without a substantial increase in computational cost.

3. **Depth and Width**: The network is 22 layers deep (considering only the layers with trainable weights) and makes use of a wider network design compared to previous architectures, yet it's computationally efficient due to smart dimensionality reduction and sparse connections.

4. **Global Average Pooling**: Unlike traditional networks that use fully connected layers at the top (which are prone to overfitting), GoogleNet employs global average pooling immediately before the classification layer, reducing the total number of parameters in the model.

5. **Auxiliary Classifiers**: To combat the problem of vanishing gradients in such a deep network, GoogleNet includes auxiliary classifiers during training. These classifiers are connected to intermediate layers and add their loss during training. Though these auxiliary networks are discarded at inference time, they help in stabilizing gradients in deeper parts of the network.

### Architecture Overview:
The GoogleNet architecture consists of multiple stacked inception modules. Each module performs convolution operations at different scales and then merges the outputs. The typical layer sequence in an inception module includes:
- 1x1 convolution (for dimension reduction),
- 3x3 convolution,
- 5x5 convolution,
- 3x3 max pooling (also followed by 1x1 convolution in some versions for dimension reduction).

This stacking and parallel operation of filters of varying sizes allow the network to adapt to various scales of input and capture intricate details from the images.

### Computational Efficiency:
Despite its depth and complexity, GoogleNet is computationally efficient. This efficiency comes from the use of 1x1 convolutions for reducing dimensionality and the sparse nature of the network where not every connection is used. This design allows GoogleNet to be deployed in environments where computational resources are limited.


## Evaluation
The models are evaluated based on their accuracy on the test set. Comparisons are drawn to understand the impact of different architectures and training enhancements like data augmentation.

## Results and Observations
- Basic CNN achieved an accuracy of approximately 72.08%.
- CNN with data augmentation achieved similar accuracy but is expected to perform better with more training epochs and GPU acceleration.
- GoogleNet model performed significantly better, achieving a test accuracy of 91%.

---

## Theoretical Background

### Convolutional Neural Networks (CNNs)
CNNs are specialized kinds of neural networks for processing data that has a grid-like topology, such as images. CNNs are composed of various layers such as convolutional layers, pooling layers, and fully connected layers that help in extracting and learning hierarchical feature representations.

### Data Augmentation
A technique to artificially expand the size of a dataset by creating modified versions of images in the dataset. This helps in training more robust models.

### Transfer Learning
Using a pre-trained model (like GoogleNet) on a new problem. This is effective because the initial layers learn universal features like edges and textures that are applicable to most imaging tasks.

### Loss Functions and Optimizers
- **Loss Functions**: Quantify the difference between the expected outcomes and predictions, e.g., Cross-Entropy Loss.
- **Optimizers**: Methods to update weights in the neural network to minimize the loss function, e.g., SGD, Adam.
