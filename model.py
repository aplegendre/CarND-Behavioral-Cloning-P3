import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

# model.py creates and trains a CNN that takes input images from the Udacity
# self-driving car simulator and outputs a steering angle
# This code is based largely on the example code shown in the SDC-ND Behavioral Cloning lesson
# The CNN is based on the NVidia paper that was provided in the lesson

# Import training image locations from training log
data_loc = 'data1'
lines = []
with open('./'+data_loc+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#del lines[0]
print(len(lines))

# Generator for providing training and validation data
# This function was copied from the Udacity generator lesson
# and adapted to include data augmentation
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            correction = 0.75 # Correction factor to steering angle output for left or right camera
            for line in batch_samples:
                measurement = float(line[3])
                # Loop through all available training images (center, right, left)
                for i in range(3):
                    source_path = line[i]
                    source_path = source_path.replace('\\', '/')
                    filename = source_path.split('/')[-1]
                    current_path = './' + data_loc + '/IMG/' + filename
                    image = cv2.imread(current_path)
                    if image != None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image) # Original training example
                        images.append(np.fliplr(image)) # Flipped example for data augmentation
                        # Steering angle output adjusted for camera position
                        if i == 1:
                            measurements.append(measurement + correction)
                            measurements.append(-(measurement + correction))
                        elif i == 2:
                            measurements.append(measurement - correction)
                            measurements.append(-(measurement - correction))

                        else:
                            measurements.append(measurement)
                            measurements.append(-measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split
# Split image locations into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2, shuffle=True)
# Define generators for training and validation examples
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Image size for input definition
keep_prob = 0.5 # Keep probability for dropout
# Model definition in Keras
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row,col,ch))) # Normalize inputs
model.add(Cropping2D(cropping=((70,25),(0,0)))) # Remove horizon and car body from images
# Remainder of network based on NVIDIA end-to-end DL for SDC paper
# Dropout added after most layer to reduce overfitting
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(keep_prob))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples), nb_epoch=7)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()