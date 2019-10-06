#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import numpy as np

#%%
import shutil
import random

# Create directory and intermediate directories
def create_dirs (dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")  

def data_classify(data_folder_path):
    create_dirs(f"{data_folder_path}/smile")
    create_dirs(f"{data_folder_path}/nonsmile")
    for i in range(1,4001):
        train = i <= 2162
        filepath = f"{data_folder_path}/file{i:04}.jpg"
        if os.path.exists(filepath):
            if train:
                shutil.move(filepath, f"{data_folder_path}/smile/file{i:04}.jpg")
                print (f"Moved {filepath} to smile folder")
            else:
                shutil.move(filepath, f"{data_folder_path}/nonsmile/file{i:04}.jpg")
                print (f"Moved {filepath} to nonsmile folder")
            # shutil.move(f"{data_folder_path}/file{i:04}.jpg", f"dataset/{"train" if train else "test"}")
        else:
            print (f"{filepath} does not exist")

def split_test_data(data_folder_path, ratio=0.2):
    create_dirs("dataset/test")
    create_dirs("dataset/train")
    idx = list(range(1,4001))
    random.shuffle(idx)
    for i, si in enumerate(idx):
        filePath = f"{data_folder_path}/file{si:04}.jpg"
        if os.path.exists(filePath):
            newFilePath = "test" if i < ratio*4000 else "train"
            newFilePath = f"dataset/{newFilePath}/file{si:04}.jpg"
            shutil.move(filePath, newFilePath)
#%%
split_test_data("../../genki4k/files", 0.2)
data_classify("dataset/test")
data_classify("dataset/train")

#%% Load data
from keras.preprocessing.image import ImageDataGenerator
# create a data generator
train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        validation_split=0.2
    )
test_datagen = ImageDataGenerator(
        horizontal_flip=True,
    )
train_it = train_datagen.flow_from_directory('dataset/train',
    class_mode='binary',
    batch_size=64,
    target_size=(192,192),
    subset='training'
    )
val_it = train_datagen.flow_from_directory('dataset/train',
    class_mode='binary',
    batch_size=64,
    target_size=(192,192),
    subset='validation'
    )
test_it = test_datagen.flow_from_directory('dataset/test',
    class_mode='binary',
    batch_size=64,
    target_size=(192,192),
    subset='training'
    )
# batchX, batchy = train_it.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#%%

# #download mnist data and split into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# #plot the first image in the dataset
# plt.imshow(X_train[0])
# #check image shape
# X_train[0].shape
# #reshape data to fit model
# X_train = X_train.reshape(60000,28,28,1)
# X_test = X_test.reshape(10000,28,28,1)
# #one-hot encode target column
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# y_train[0]

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(192,192,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#train model
#%%
# model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)
model.fit_generator(train_it, steps_per_epoch=40, validation_data=val_it, validation_steps=10)
loss = model.evaluate_generator(test_it, steps=24)
#show predictions for the first 3 images in the test set
# model.predict(X_test[:4])
# #show actual results for the first 3 images in the test set
# y_test[:4]

#%%
