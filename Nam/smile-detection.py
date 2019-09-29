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
# Create directory and intermediate directories
def create_dirs (dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")  

def data_preprocessing(data_folder_path):
    import shutil
    create_dirs("dataset/smile")
    create_dirs("dataset/nonsmile")
    for i in range(1,4001):
        train = i <= 2162
        filepath = f"{data_folder_path}/file{i:04}.jpg"
        if os.path.exists(filepath):
            if train:
                shutil.move(filepath, f"dataset/smile/file{i:04}.jpg")
                print (f"Moved {filepath} to smile folder")
            else:
                shutil.move(filepath, f"dataset/nonsmile/file{i:04}.jpg")
                print (f"Moved {filepath} to nonsmile folder")
            # shutil.move(f"{data_folder_path}/file{i:04}.jpg", f"dataset/{"train" if train else "test"}")
        else:
            print (f"{filepath} does not exist")

data_preprocessing("../genki4k/files")

#%% Load data
from keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory('dataset', class_mode='binary', batch_size=32)
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#%%

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plot the first image in the dataset
plt.imshow(X_train[0])
#check image shape
X_train[0].shape
#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]
#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)
#show predictions for the first 3 images in the test set
model.predict(X_test[:4])
#show actual results for the first 3 images in the test set
y_test[:4]