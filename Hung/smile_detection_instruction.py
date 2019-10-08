# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:04:52 2019

@author: Hung
"""

#%% Load libraries

import numpy as np
import os
import cv2
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#%% Load data

# Load label
with open('labels.txt','r') as f:
    labels = f.readlines()
    
y = [int(line.split(' ')[0]) for line in labels]
#lb = LabelBinarizer()
#y = lb.fit_transform(y)

# Load images
X = []
filenames = os.listdir('files/')
for name in filenames:
    X.append(pyplot.imread('files/' + name))

X = [cv2.resize(image,(64,64)).astype(np.uint8) for image in X]

# Check if there is grayscale image. If there is, convert to RGB
for i in range(len(X)):
    if X[i].shape != (64,64,3):
        X[i] = cv2.cvtColor(X[i],cv2.COLOR_GRAY2RGB)

X = np.array(X)

# Normalize
X = X.astype(float) / 255.0

#%% Train and test split 80/20

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size= 0.2,
                                                 random_state=0)

#%% Create model

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_uniform', input_shape = X_train.shape[1:]))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_uniform'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_uniform'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_initializer= 'he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(1,activation = 'sigmoid', kernel_initializer= 'he_uniform'))

model.summary()

model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

#%% Train

hist = model.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test))