from skimage import img_as_ubyte
from skimage import exposure
from skimage.transform import resize
from scipy.misc import imresize

def brightness(image):
    """
    Adjust the brightness of the image randomly
    in the gain range of 0.3-0.6.
    :image: an ndarray image
    :output: an ndarray image that is brightness adjusted
    """
    gain = (np.random.randint(3)+3)/10
    return exposure.adjust_gamma(image, gamma=1.0, gain=gain)

def my_resize(image):
    """
    Resizing the image to (128,128,3)
    :image: an ndarray image
    :output: an ndarray image that is resized
    """
    return imresize(image, size=(128,128,3))

def pix_val(im):
    """
    Cast a vertical shadow on image, randomly reduce intensity of
    all rgb channels by 70-120.
    :im: an ndarray image
    :output: an ndarray image that its brightness is 
    partially adjusted
    """
    cons = (np.random.randint(6)+7)*10
    for idx,val in enumerate(im):
        for idx2,val2 in enumerate(val):
            for idx3,val3 in enumerate(val2):
                tmp = val3-cons
                if tmp < 0:
                    im[idx,idx2,idx3] = 0
                else:
                    im[idx,idx2,idx3] = tmp
    return im

def random_shadow(image):
    """
    To random which part of the image the shadow will cast on
    :image: an ndarray image
    :output: an ndarray image that its brightness is 
    partially adjusted
    """
    lo_bound = np.random.randint(0,88)
    hi_bound = lo_bound + 40
    new_img = image.copy()
    new_img[:,lo_bound:hi_bound] = pix_val(new_img[:,lo_bound:hi_bound])
    return new_img

def flipping(image):
    """
    Flip the image along the horizontal axis
    :image: an ndarray image
    :output: an ndarray image that is flipped
    """
    return np.fliplr(image)

def process_image(img):
    """
    Flip a coin and decide on whether the image will be 
    brightness adjusted or cast a shadow
    :image: an ndarray image
    :output: an ndarray image that is processed
    """
    resized = my_resize(img)
    if np.random.randint(2) == 1:
        processed_img = brightness(resized)
    else:
        processed_img = random_shadow(resized)
    return processed_img

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

import os
import csv
from sklearn.utils import shuffle

# read the data in, to use with generator()
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
shuffle(samples)

# Split the Train/Validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn
from sklearn.utils import shuffle
from tqdm import tqdm
from PIL import Image

# HERE I HAVE 2 Generators
# 1. for training set which do real-time augmentation
# 2. for validation set which to validate the model by feeding only center-camera image

def generator(samples, batch_size=32):
    """
    Generator for Training dataset Doing data augmentation by
    1. Adjusting the steering correction of left/right image
    2. Pre-Process left/center/right images by brightness adjustment or shadown casting
    3. Flip the images along the horizontal-axis
    4. Pre-Process the flipped images
    :sample: a list of rows read from driving_log.csv
    :batch_size: the number of batch
    :output: a tuple of (images, steering angle)
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                
                if (steering_center == 0.0 and round(np.random.randint(10)) == 0) or steering_center != 0.0:

                    # create adjusted steering measurements for the side camera images
                    correction = 0.2 # this is a parameter to tune
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    # read in images from center, left and right cameras
                    path = "data/" # fill in the path to your training IMG directory
                    img_center = my_resize(np.asarray(Image.open(path + batch_sample[0].strip())))
                    img_left = my_resize(np.asarray(Image.open(path + batch_sample[1].strip())))
                    img_right = my_resize(np.asarray(Image.open(path + batch_sample[2].strip())))

                    processed_img_center = process_image(img_center)
                    processed_img_left = process_image(img_left)
                    processed_img_right = process_image(img_right)

                    # add images and angles to data set
                    car_images.extend([img_center, img_left, img_right])
                    steering_angles.extend([steering_center, steering_left, steering_right])

                    car_images.extend([processed_img_center, processed_img_left, processed_img_right])
                    steering_angles.extend([steering_center, steering_left, steering_right])

                    # flip the image
                    img_center_flip = flipping(img_center)
                    img_left_flip = flipping(img_left)
                    img_right_flip = flipping(img_right)

                    processed_img_center_flip = flipping(processed_img_center)
                    processed_img_left_flip = flipping(processed_img_left)
                    processed_img_right_flip = flipping(processed_img_right)

                    # flip the angle
                    steering_center_flip = -steering_center
                    steering_left_flip = -steering_left
                    steering_right_flip = -steering_right

                    # Append to the list
                    car_images.extend([img_center_flip, img_left_flip, img_right_flip])
                    steering_angles.extend([steering_center_flip, steering_left_flip, steering_right_flip])

                    car_images.extend([processed_img_center_flip, processed_img_left_flip, processed_img_right_flip])
                    steering_angles.extend([steering_center_flip, steering_left_flip, steering_right_flip])
                    
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

            
# just like the train generator() but get only the center image and steering center
def valid_generator(samples, batch_size=32):
    """
    Generator for validation dataset just like the generator() 
    but get only the center image and steering center
    :sample: a list of rows read from driving_log.csv
    :batch_size: the number of batch
    :output: a tuple of (images, steering angle)
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                
                # read in images from center, left and right cameras
                path = "data/" # fill in the path to your training IMG directory
                img_center = np.asarray(Image.open(path + batch_sample[0].strip()))
                resized = my_resize(img_center)
                # add images and angles to data set
                images.extend([resized])
                angles.extend([steering_center])
                    
            X_valid = np.array(images)
            y_valid = np.array(angles)
            yield sklearn.utils.shuffle(X_valid, y_valid)

            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = valid_generator(validation_samples, batch_size=32)

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(128,128,3,), name='input_normalization'))

# Cropping the image 
model.add(Cropping2D(cropping=((50,20), (0,0)), name='top:50px/bottom:20px'))

model.add(Convolution2D(24, 5, 5, subsample = (2,2), name='24Aconv5x5/stride2'))
model.add(Activation('elu', name='ELU_1'))

model.add(Convolution2D(36, 5, 5, subsample = (2,2), name='36Aconv5x5/stride2'))
model.add(Activation('elu', name='ELU_2'))

model.add(Convolution2D(48, 5, 5, subsample = (2,2), name='48Aconv5x5/stride2'))
model.add(Activation('elu', name='ELU_3'))

model.add(Convolution2D(64, 3, 3, name='64Aconv5x5/stride1'))
model.add(Activation('elu', name='ELU_4'))

model.add(Flatten(name='Flat'))

model.add(Dense(1000,name='1000-FC'))
model.add(Activation('elu', name='ELU_5'))

model.add(Dropout(0.5, name='DROPOUT_1_p0.5'))

model.add(Dense(100,name='100-FC'))
model.add(Activation('elu', name='ELU_6'))

model.add(Dropout(0.5, name='DROPOUT_2_p0.5'))

model.add(Dense(50, name='50-FC'))
model.add(Activation('elu', name='ELU_7'))

model.add(Dropout(0.5, name='DROPOUT_3_p0.5'))

model.add(Dense(1, name='1-FC'))

# Compile and Run the model
model.summary()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples), validation_data=validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=60, verbose=1)

model.save('model.h5')