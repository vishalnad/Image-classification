from __future__ import print_function
import keras
import re
import os
import random
import cv2
from numpy import array
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam


batch_size = 32
num_classes = 100
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

#Extracting paths of all images in dataset along with labels
dir = '/home/vishalmn/Omnamahshivaay/IMFDB_final'                                                                                                                                                                                                          
images = []
labels = []
for root, dirs, files in os.walk(dir):
    for name in files:
        if name.endswith('.jpg'):
            img = os.path.join(root, name)
            images.append(img)
            found = re.search('.*IMFDB_final/(.+?)/.*',root).group(1)
            labels.append(found)
            #print(found)
            
            
image_dataset = []

#Loading images from the dataset
for image in images:
    img = cv2.imread(image)
    img = cv2.resize(img, (img_rows, img_cols))
    img = img_to_array(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_rows, img_cols))
    #img = img.reshape((img_rows,img_cols,1))
    image_dataset.append(img)
print("Successfully read images")

#Shuffling the dataset
combine = list(zip(image_dataset,labels))
random.shuffle(combine)
image_dataset[:],labels[:] = zip(*combine)

#Converting list to numpy array for feeding into model 
image_dataset = np.array(image_dataset, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(x_train, x_test, y_train, y_test) = train_test_split(image_dataset,
    labels, test_size=0.2, random_state=42)

print('Datasets ',image_dataset.shape)
print('x_train shape : ', x_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Creating model and stacking up layers
model = Sequential()
#Layer 1, ie input layer
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
#Layer 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#Layer 3
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#Layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#Layer 5
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
#Layer 6, ie output layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#Execution
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

#Final results
print('Test loss:', score[0])
print('Test accuracy:', score[1])

