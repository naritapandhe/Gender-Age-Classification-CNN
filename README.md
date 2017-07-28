##########################################################
# Gender and Age Classification using CNNs
##########################################################
## Overview
This is my final course project for CSCI 8360 Data Science Practicum course. This project focuses on gender and age classification based on images. My work builds on the previous work of:

```
 Gil Levi & Tal Hassner Alexander, Age and Gender Classification using Convolutional Neural Networks. 2015
 Ari Ekmekji. Convolutional Neural Networks for Age and Gender Classification, Stanford University. 2016
 ```
 
Both of these papers have established efficient architectures for solving gender and age classification problem. I've tried to extend their approach in order to improve the results. The primary area of experimentation is to tweak previously published architecture in terms of depth of the network, number of parameters in the network, modifications to parameters of the network or the layout of these networks. I've tried to chain the architectures for age and gender classification to take advantage of gender-specific age characteristics inherent to images.

The dataset used for training and testing for this project is the [Adience Benchmark - collection of unfiltered face images](http://www.openu.ac.il/home/hassner/Adience/data.html). It contains total 26,580 images of 2,284 unique subjects that are collected from Flickr [10]. There are 2 possible gender labels: M, F and 8 possible age ranges: 0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+. Each image is labelled with the person’s gender and age-range (out of 8 possible ranges mentioned above). From the original dataset I've used mostly frontal face images reducing the dataset size to 17,523 images. The images are subject to occlusion, blur, reflecting real-world circumstances. 

![sample](https://cloud.githubusercontent.com/assets/3252684/21166797/c6599684-c175-11e6-9714-8125febf14dc.png)

##########################################################
## Preprocessing
The following preprocessing was applied to each image:

- Have trained the network on frontal faces images
- Random crops of 227 × 227 pixels from the input image of size 256 × 256 
- Randomly mirror images in each forward-backward training pass
- When predicting, the network expects face image, cropped to 227 × 227 around the face center.


##########################################################
## Model Description
For **Gender Classification**, following are the details of the model: 

1. 3x7x7 filter shape, 96 feature maps. Stride of 4 and 0 padding. Followed by: ReLU, Max-Pool, LRN
2. 96x28x28 filter shape, 256 feature maps. Followed by: ReLU, Max-Pool, LRN
3. 256x3x3 filter shape, stride 1 and padding 1. ReLU, Max-Pool. 
4. Fully connected layer of 512 neurons. Followed by : ReLU, Dropout = 0.5. 
5. Fully connected layer of 512 neurons. Followed by : ReLU, Dropout = 0.5. 
6. Last layer maps to the 2 classes for gender
  
Since, gender and age classification has been chained i.e. based on gender, classify age, 2 separate age classifiers: Male-Age and Female-Age classifiers have been built. Based on the results of gender classification, the images are fed to the respective gender-based age classifiers.  

For gender-based **Age Classification**, same model(Gender Classification model) has been used with the following modifications:

1. Dropouts in the second fully connected layer have been modified to be 0.7
2. Addition of weighted losses
3. Last layer maps to the 8 classes

For both Age and Gender classification, training is performed using Stochastic Gradient Descent having a batch size of 50. The initial learning rate is 1e−3, reduced to 5e-4 after every 10,000 iterations. The models have been trained using 4-fold cross validation.

##########################################################
## Instructions for Running the Model

Ensure the following Python packages are installed on your machine:

* numpy
* tensorflow 
* sklearn
* scipy 
* pandas
* matplotlib

Once your environment has been setup, download the project files and run the following:

1. For gender classification execute: python gender/train_n_test_gender.py

 The script expects: Path to training and testing data.
 Cross validation accuracy is recorded every 1000 iterations. Predictions are saved every 1000 iterations to predicted_genders.txt

2. Based on the gender classification results, to separate out the data of predicted males and females, execute the script: gender/create_gender_test_based_on_predictions.py

3. Once we have separated out the predicted males and females, we can then feed them gender-based age classifiers to get the age. Inorder to do so, execute the script: 
  1. python age/train_n_test_male_model.py   *#Execute this for predicted males*
  2. python age/train_n_test_female_model.py  *#Execute this for predicted females*
  For both the scripts, cross validation accuracy is recorded every 1000 iterations. Predictions are saved every 1000 iterations to predicted_X_age_prediction.txt  *X -> can be either males or females*

##########################################################
## Results
All the results have been logged here: https://github.com/eds-uga/gender-age-classification/issues/3


 
 
 
