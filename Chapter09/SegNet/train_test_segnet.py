"""This module implements training and testing the SegNet model."""
from __future__ import absolute_import
from __future__ import print_function

import pylab
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import cv2

from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, ZeroPadding2D
from keras.layers import BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

import keras
keras.backend.set_image_dim_ordering('th')

from tqdm import tqdm
import itertools


# define optimizer
optimizer = Adam(lr=0.002)
# input shape to the model
input_shape=(3, 360, 480)
# training batchsize
batch_size = 6
# number of training epochs
nb_epoch = 60

annFile='annotations/annotations/instances_train2014.json'
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

def data_list(imgIds, count = 12127, ratio = 0.2):
    """Function to load image and its target into memory."""
    img_lst = []
    lab_lst = []

    for x in tqdm(imgIds[0:count]):
        # load image details
        img = coco.loadImgs(x)[0]

        # read image
        I = io.imread('images/train2014/'+img['file_name'])
        if len(I.shape)<3:
            continue

        # load annotation information
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

        # load annotation
        anns = coco.loadAnns(annIds)

        # prepare mask
        mask = coco.annToMask(anns[0])

        # This condition makes sure that we select images having only one person
        if len(np.unique(mask)) == 2:

            # Next condition selects images where ratio of area covered by the
            # person to the entire image is greater than the ratio paramater
            # This is done to not have large class imbalance
            if (len(np.where(mask>0)[0])/len(np.where(mask>=0)[0])) > ratio :

                # If you check, generated mask will have 2 classes i.e 0 and 2
                # (0 - background/other, 1 - person).
                # to avoid issues with cv2 during the resize operation
                # set label 2 to 1, making label 1 as the person.
                mask[mask==2] = 1

                # resize image and mask to shape (480, 360)
                I= cv2.resize(I, (480,360))
                mask = cv2.resize(mask, (480,360))

                # append mask and image to their lists
                img_lst.append(I)
                lab_lst.append(mask)
    return (img_lst, lab_lst)

img_lst, lab_lst = data_list(imgIds)

print('Sum of images for training, validation and testing :', len(img_lst))
print('Unique values in the labels array :', np.unique(lab_lst[0]))

def make_normalize(img):
    """Function to histogram normalize images."""
    norm_img = np.zeros((img.shape[0], img.shape[1], 3),np.float32)

    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]

    norm_img[:,:,0]=cv2.equalizeHist(b)
    norm_img[:,:,1]=cv2.equalizeHist(g)
    norm_img[:,:,2]=cv2.equalizeHist(r)

    return norm_img

def make_target(labels):
    """Function to one hot encode targets."""
    x = np.zeros([360,480,2])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def model_data(images, labels):
    """Function to perform normalize and encode operation on each image."""
    # empty label and image list
    array_lst = []
    label_lst=[]

    # apply normalize function on each image and encoding function on each label
    for x,y in tqdm(zip(images, labels)):
        array_lst.append(np.rollaxis(make_normalize(x), 2))
        label_lst.append(make_target(y))

    return np.array(array_lst), np.array(label_lst)

# Get model data
train_data, train_lab = model_data(img_lst, lab_lst)

flat_image_shape = 360*480

# reshape target array
train_label = np.reshape(train_lab,(-1,flat_image_shape,2))

# test data
test_data = train_data[1900:]
# validation data
val_data = train_data[1500:1900]
# train data
train_data = train_data[:1500]

# test label
test_label = train_label[1900:]
# validation label
val_label = train_label[1500:1900]
# train label
train_label = train_label[:1500]


model = Sequential()
# encoding
model.add(Layer(input_shape=input_shape))
model.add(ZeroPadding2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())

# decoding
model.add(ZeroPadding2D())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2)))

model.add(ZeroPadding2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=2, kernel_size=(1, 1), padding='valid'))

model.add(Reshape((2,flat_image_shape)))
model.add(Permute((2, 1)))
model.add(Activation('softmax'))


# compile model
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.002), metrics=["accuracy"])
# use ReduceLROnPlateau to adjust the learning rate
reduceLROnPlat = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1)
callbacks_list = [reduceLROnPlat]

# fit/train the model
history = model.fit(train_data, train_label, callbacks=callbacks_list,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, shuffle=True,
                    validation_data=(val_data, val_label))

# test
loss,acc = model.evaluate(test_data, test_label)
print('Loss :', loss)
print('Accuracy :', acc)
