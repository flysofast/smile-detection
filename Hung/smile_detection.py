# -*- coding: utf-8 -*-

#%% Load libraries

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#%% Utils

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    This function is borrowed from a course of Prof. Heikki Huttunen
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#%% Load data

# Load label
with open('labels.txt','r') as f:
    labels = f.readlines()
    
y = np.array([int(line.split(' ')[0]) for line in labels])
nb_classes = 2
y = to_categorical(y, nb_classes).astype(np.float32)

# Load images
X = []
filenames = os.listdir('files/')
for name in filenames:
    X.append(plt.imread('files/' + name))

#%% Pre-processing
# Resize image to 64x64
X = [cv2.resize(image,(64,64)).astype(np.uint8) for image in X]

# Convert to grayscale
for i in range(len(X)):
    if len(X[i].shape) == 3:
        X[i] = cv2.cvtColor(X[i],cv2.COLOR_RGB2GRAY)

X = np.array(X)

# Normalize
X = X.astype(float) / 255.0

# Add third dimension to images
X = np.expand_dims(X, axis=3)

# class weights
nb_sample_per_class = y.sum(axis=0)
class_weights = nb_sample_per_class.max()/nb_sample_per_class

#%% Train and test split 80/20

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size= 0.2,
                                                 random_state=0)

#%% Create model

model = Sequential()
model.add(Conv2D(16, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal', input_shape = X_train.shape[1:]))
model.add(Conv2D(16, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal'))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer= 'he_normal'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = 'relu', kernel_initializer= 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(2,activation = 'softmax'))

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#%% Train
hist = model.fit(X_train, y_train, 
                 epochs = 100, 
                 class_weight=class_weights, 
                 validation_data = (X_test, y_test))

#%% Test

y_prob = model.predict(X_test)

y_pred = np.argmax(y_prob, axis = 1)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Accuracy: {:.2f} %".format(100*accuracy))

#%% Plot accuracy, loss and cofusion matrix
# Accuracy
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Confusion matrix
plt.figure()
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=['no smile','smile'],
                      title='Confusion matrix')
    
#%% Save model
model.save("model.h5")