##########################################################
# Gender and Age Classification using CNNs
##########################################################
## Overview
---
This is my final course projects for CSCI 8360 Data Science Practicum course. This project focuses on gender and age classification based on images. My work builds on the previous work:

 -*Gil Levi & Tal Hassner Alexander, Age and Gender Classification using Convolutional Neural Networks. 2015*
 -*Ari Ekmekji. Convolutional Neural Networks for Age and Gender Classification, Stanford University. 2016*
 
Both of these papers have established efficient architecture for solving gender and age classification problem. I've tried to extend their approach in order to improve the results. The primary area of experimentation is to tweak previously published architecture in terms of depth of the network, number of parameters in the network, modifications to parameters of the network or the layout of these networks. I've tried to chain the architectures for age and gender classification to take advantage of gender-specific age characteristics inherent to images.

The dataset used for training and testing for this project is the [Adience Benchmark - collection of unfiltered face images](http://www.openu.ac.il/home/hassner/Adience/data.html). It contains total 26,580 images of 2,284 unique subjects that are collected from Flickr [10]. There are 2 possible gender labels: M, F and 8 possible age ranges: 0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+. Each image is labelled with the person’s gender and age-range (out of 8 possible ranges mentioned above). From the original dataset I used mostly frontal face images reducing the dataset size to 17,523 images. The images are subject to occlusion, blur, reflecting real-world circumstances. 

![sample](https://cloud.githubusercontent.com/assets/3252684/21166797/c6599684-c175-11e6-9714-8125febf14dc.png)

##########################################################
## Preprocessing
---
The following preprocessing was applied to each image:

- Have trained the network on frontal faces images
- Take a random crop of 227 × 227 pixels from the input image of size 256 × 256 
- Randomly mirror it in each forward-backward training pass
- When predicting, the network expects face image, cropped to 227 × 227 around the face center.


##########################################################
## Model Description
---
For Gender Classification, following are the details of the model: 
1. 96 filters of size 3x7x7 pixels are applied to the input with a stride of 4 and 0 padding. This is followed by a rectified linear operator (ReLU), a Max-Pooling layer taking the maximal value of 3x3 regions with two-pixel strides and a local response normalization layer.
2. Second convolutional layer then processes the 96x28x28 output of the previous layer with 256 filters of size 96x5x5. Followed by ReLU, Max-Pooling layer and LRN
3. In the third layer, 384 filters of size 256x3x3 are convolved with stride 1 and padding 1, followed by a ReLU and Max-Pool. 
4. The fully connected layers:
 





 
 
 
