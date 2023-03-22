import sys
import os
import cv2
import numpy as np
from utils import DatasetLoad
# import the necessary packages
from keras.models import load_model
# we know our label are cat, dog, and panda
label = ['cat', 'dog', 'panda']
# Resize size
width = 64
height = 64

print("[INFO] loading trained network...")
model = load_model('lenet_medel.hdf5')
print('[INFO] Call some image to test ...')
path = 'datasets/animals/dogs'
listfiles = os.listdir(path)
for (i, imagefile) in enumerate(listfiles):
    imagepath = path+'/'+imagefile
    image = cv2.imread(imagepath)
    ing = image.copy()
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = img.astype("float") / 255.0
    img = img.reshape(1,width,height,3)
    pred = model.predict(img)
    print(pred)
# Find the index of maximum predition
ind_predmax = np.argmax(pred)

cv2.putText(image, "Label: 1".format(label[ind_predmax]),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
cv2.imshow('Org',image)
key = cv2.waitKey(1000)&0xFF
if key == ord('q'):
    break

