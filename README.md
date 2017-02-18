# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[architect]: /report_img/model_architecture.png "Model Visualization"
[explo1]: /report_img/data_exploration1.png "Data Exploration: Before Augmentation"
[explo2]: /report_img/data_exploration2.png "Data Exploration: After Augmentation"
[augment]: /report_img/data_augment1.png "Data Augmentation"
[train_valid]: /report_img/train_valid.png "Train/Validate loss"


Overview
---

To meet specifications, the project will require submitting four files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)

Optionally, a video of your vehicle's performance can also be submitted with the project although this is optional. This README file describes how to output the video in the "Details About Files In This Directory" section.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

### `report_img/*.png`
Contains all the images used in writeup_report.md & README.md

### `behavioral_cloning.ipynb`
The notebook that use for visualization and experiment

### `model.py`
The file that use with Udacity's dataset and export `model.h5` to be used 
with `drive.py`

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The strategy used for deriving an architecture was to make the network that could
predict the angle solely from an image.

I did not start building the architecture from scratch. In fact, I used a cnn the base network as
in the [nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The main reasons I thought this model was appropriate are that:
1. It combined many convolutional layers with appropriate sizes for feature engineering.
2. It was proved by visualization of the activation of feature maps in the paper that the architecture is 
able to learn useful features on its own.

To test how cool my model was, I splitted the dataset (image, steering_angle) into training/validation set.

The loss of my model was too high, that i decided to work more on data augmentation & tuning left/right camera offset.

Moreover, I add [dropout layers to prevent complex co-adaptation](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) & overfitting 
between each of last 3 Fully Connected layers

The final step was to run the simulator to see how well the car was driving around track one. At first, the car always went off the track.
I notice this bahavior as it does not know how to get back to the center of lane.

To improve the driving behavior in these cases, I fine-tuned the left/right camera offset.
I found that 0.2 was an acceptable number which I could let the car drive for hours without going off track despite some jittery.

#### 2. Final Model Architecture

Here is a (long) visualization of the architecture

![alternate][architect]

My model can be describe in details by the following list (on model.py lines xx-xx) :

 1. Lambda layer - for normalization
 1. Cropping Layer - crop 50px from the top, 20px from the bottom
 1. 24@5x5 stride-2 convolutional layer following by ELU Activation
 1. 36@5x5 stride-2 convolutional layer following by ELU Activation
 1. 48@5x5 stride-2 convolutional layer following by ELU Activation
 1. 64@3x3 stride-1 convolutional layer following by ELU Activation
 1. Flatten layer
 1. 1000 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 100 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 50 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 1 Fully Connected layer

#### 3. Creation of the Training Set & Training Process


To create a training set, I am more interested in using
udacity's provided dataset than collecting my own dataset 
because of the following reasons
 - the dataset that udacity provides are small, left-biased, and 0-angles biased.
 Thus, it can't be used directly.
 Trying to make it work involves much effort on understanding the data, task, 
 and architecture which I found challenging.
 - the udacity's task state that to recover the car from off-center to center can be
 done by collecting data which a car drives from the corner to the center, but I like
 how nvidia works on the offset adjustment for left/right camera images which is perfect
 for me to make the most out of existing data.

To get myself into udacity's data, I first explore the data

![alt text][explo1]
*The histrogram of the udacity's dataset: the X-axis is the steering_angle 
and the Y-axis is the number that element on X-axis occurs (plot with bin=50)*

>I noticed some biased on the 0 angle, so I know that I need to balance the data.
Meanwhile I perceived the numbers of data for other angles are too small.
Data Augmentation is definitely needed here

![alt text][augment]

My Augmentation set here is:
 1. Offset images from left/center/right camera
    - additional off-center shifts of images are then used 
    for network to learn how to recover from mistakes or it will 
    slowly drive off the center, the magnitude of 
    shift value seem to be 0.2. I tried(0.08, 0.1, 0.2) the last one was the best,
    but I'm sure it can be better
 1. Resize the images to (128,128)
    - the numbers are from eyeballing. I tried 64,64 but I think there must 
  be loss of information, so I try larger image sizes that my eyes & 
  training speed feel comfortable with
 1. Flip
     - both the image along the horizontal-axis and angle
 1. Random brightness **or** shadow on both flipped and non-flipped images
     - brightness adjustment 
         - by adjusting gain in the range of 0.3-0.6
     - shadows shade
         - random vertical line with the width of 40 pixels
 1. Cropping
     - the top for 50 pixels
     - the bottom for 20 pixels
 1. Randomly select only 10% of 0.0 angles dataset available

Here is the result of after implementing data augmentation pipeline, there are much more 
data available to train from ~8k to ~50k

![alt text][explo2]

However, the data is too big to fit in the memory, so I use Keras's generator() to
do data real-time augmentation while training the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 60 as evidenced by it seems that another 10 epochs
might make the network overfitted.

![alt text][train_valid]

I used an adam optimizer, so no learning rate was adjusted.