# **Behavioral Cloning** 

## Writeup for Project 3

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_network.PNG "NVIDIA Network"
[image2]: ./examples/center_lane.jpg "Center Lane Driving"
[image3]: ./examples/inside_corner.jpg "Hugging the Inside Corner"
[image4]: ./examples/turn_exit.jpg "Exiting a Turn"
[image5]: ./examples/not_flipped.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (not materially changed from original)
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used is based on the NVIDIA End-to-End Learning for Self-Driving Cars paper (arXiv:1604.07316v1) that was recommend and demonstrated in the lesson for this project.

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 79-87). It has fully connected layers with depths between 10 and 100 (model.py lines 89-93). The output layer has a depth of 1, since this is a regression network and the only output is steering angle.

The model includes RELU layers to introduce nonlinearity for all convolutional layers, and the data is normalized in the model using a Keras lambda layer (model.py line 75). Images are cropped using a Keras cropping layer (model.py line 79) so that both training and test images are cropped in the same way. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80, 81, 84, 86, 90). Dropout with a 50% keep probability is used after most convolutional layers and after the first fully connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. These datasets included forward and backward drives on both track types.

Data augementation was also used to try to reduce overfitting (model.py lines 36-59). The training data was augmented by flipping each examples from left to right and inverting the steering angle. It was also augmented by using data from the left and right cameras and the applying a correction factor for steering angle. The left and right camera data was flipped using the same method as the center camera data. This resulted in 6x more examples than the baseline training set from only the center camera.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried to keep my driving in the center of the lane, but would often drift to the side. This meant that the data had both recovery examples and examples of drifting out of the center, which resulted in slightly erractic driving on straightaways. None of my examples had the car driving off of the road and they represented my best attempt at optimal driving.

For more detail on my training data, see the training set section below.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy an existing successful architecture, add data augmentation, and then try to achieve good performance by providing appropriate training examples. I added dropout to this model in order to proactively avoid overfitting. I included dropout after the convolutional layers and the first fully connected layer because those were dropout locations that were effective for me in the traffic sign classifier project.

My first step was to use a convolution neural network model similar to the NVIDIA model that was shown in the lesson for this project. I thought this model might be appropriate because it had already been successfully used to train a real self-driving car in a more complex environment.

I also started by using the data augmentations that were recommended in the lesson, which included both flipping the training set and including the left and right camera images with a steering angle correction factor.

In order to gauge how well the model was working, I started with the example data set that was included in the project. The model originally drove straight off of the track because my initial correction factor for the left and right camera images was too high. I then focused on adjusting the correction factor until the model could drive a significant portion of the track.

Once the correction factor was tuned, the trained model was able to drive the car up until the dirt track detour section of track one. The lane in this section had a different marking than the other sections, so I assumed that this was due to a lack of sufficient training data. I therefore shifted my focus to improving the training set as described in the section below.

With the improved training set, the model was able to drive autonomously around track one without falling off. The new training set decreased the smoothness of straightaway driving, but was able to make it all the way around the track, unlike the original training data.

I also evaluated the model on track two, since I had included that track in the training data. It performs well in many sections, but would struggle around cliff sections where the lane was marked by signs rather than road features. This lower performance might be due to over-cropping of the input images, so I may try it again with various cropping sizes in the future.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-93) consisted of a convolution neural network with the following layers and layer sizes:


| Layer         		|     Description	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 160x320x3 RGB image   								| 
| Normalization			| Lambda layer that brings all values between 0 and 1	|
| Cropping				| Removes top 70 and bottom 25 rows of pixels			|
| Convolution 5x5		| Relu activation, 2x2 subsampling, output depth of 24 	|
| Dropout				| 50% keep probability									|
| Convolution 5x5		| Relu activation, 2x2 subsampling, output depth of 36 	|
| Dropout				| 50% keep probability									|
| Convolution 5x5		| Relu activation, 2x2 subsampling, output depth of 48 	|
| Dropout				| 50% keep probability									|
| Convolution 3x3		| Relu activation, no subsampling, output depth of 64 	|
| Dropout				| 50% keep probability									|
| Convolution 3x3		| Relu activation, no subsampling, output depth of 64 	|
| Fully connected		| Output depth of 100									|
| Dropout				| 50% keep probability									|
| Fully connected		| Output depth of 50									|
| Fully connected		| Output depth of 10									|
| Fully connected		| Output depth of 1, Final Steering Output				|


Here is a visualization of the architecture from the NVIDIA End-to-End Learning for Self-Driving Cars paper (arXiv:1604.07316v1) that this model was based on:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using my best attempt at center lane driving and then once again backwards. In some cases, I would accidently over- or understeer, which resulted in erractic straightaway performance. Here is an example image of center lane driving:

![alt text][image2]

When driving around corners or S-curves, I would try to stay close to the inside corner, which would reduce the risk of understeering. This resulted in better handling of turns, but may have contributed to the degraded straightaway performance. Here is an example of the training data that follows the inside curve:

![alt text][image3]

This driving method for turns also provided some information for recovery data. When exiting the turns, the car would not be in the center of the lane and would need to find its way back to the center. Since recovery was naturally part of my data, I did not record specific recovery laps. Here is an example of the car exiting a turn off-center:

![alt text][image4]

Then I repeated this process on track two in order to get more data points and generalize the model. I only drove once in each direction on track two, rather than twice forward as I had done on track one. I did this so that there would be more examples for the required test condition, i.e. forward driving on track one.

To augment the data sat, I added images from the left and right cameras and added an offset to their steering angle output. I also flipped all of the images and angles in order to reduce the impact of the left turn bias of each track. By flipping images, I had an equal number of examples with left and right turns. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 9446 examples of images and steering angle pairs. 

I considered preprocessing the images beyond the normalization and cropping that was in my network, but it seemed to degrade performance. The preprocessing that I tried was histogram normalization to improve track two performance in shadowed sections, but eliminated it because it hurt my track one performance.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by leveling off of validation accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
