
import sys
import os
import cv2
import joblib
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import DatasetLoad
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())
pathes = args["dataset"]

# Resize size
width = 64
height = 64

# Initial load dataset with resize
data = DatasetLoad(width,height)

# Load dataset from paths
print('[INFO] loading datasets...')

# we know our label are cat, dog, and panda
label = ['cat', 'dog', 'panda']
classes = len(label)

# verbose = 500. it means we want to show on screen only when it achieve 500 images read
datas, labels = data.load(pathes, verbose = 500)
datas = datas.astype("float") / 255.0
print('[INFO] shape of datas = ', datas.shape)
print('[INFO] split dataset to training and testing dataset ...')
(trainX, testX, trainY, testY) = train_test_split(datas, labels, test_size=0.20, random_state=42)

# encode the labels as binary
print(trainY)
print(testY)
le = LabelBinarizer()
trainY = le.fit_transform(trainY) # we convert labels to binary number
testY = le.fit_transform(testY) # we convert labels to binary number
print(trainY)
print(testY)

# initialize the optimizer and model
print("[INFO] Initialize the optimizer and model...")
opt = SGD(Ir=0.01)
# initialize the model
model = Sequential()
inputShape = (width, height,3)
# first set of CONV => RELU => POOL layers
model.add(Conv2D(32,(3,3),padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# second set of CONV => RELU => POOL layers
model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))
# Mulitple classes, normally we use categorical_crossentropy loss function
# For only two classes, we can use binary_crossentropy loss funciton
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
# batch size = 1, mean we train one image per one iteration
# epochs is once we finish training whole dataset
# epochs = 20, mean we do looping till 20 times of training whole dataset
batches =16
epochs = 25
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batches, epochs=epochs, verbose=1)
# Save model
model.save('lenet_medel.hdf5',overwrite=True)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batches)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
target_names=[str(x) for x in le.classes_]))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs ), H.history["loss"],label="train_loss")
plt.plot(np.arange(0, epochs ), H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,epochs), H.history["acc"],label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()