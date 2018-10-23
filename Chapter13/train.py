"""This module imports other modules to train the vgg16 model."""
from __future__ import print_function

from crop_resize_transform import model_data
from test import test

import matplotlib.pyplot as plt

import random
from scipy.io import loadmat
import numpy as np
import pandas as pd
import cv2 as cv
import glob

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
from keras import applications
K.clear_session()


# set seed for reproducibility
seed_val = 9000
np.random.seed(seed_val)
random.seed(seed_val)

# load the examples file
examples = loadmat('FLIC-full/examples.mat')
# reshape the examples array
examples = examples['examples'].reshape(-1,)

# each coordinate corresponds to the the below listed body joints/locations
# in the same order
joint_labels = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                'mllleg', 'mrlleg']

# print list of known joints
known_joints = [x for i, x in enumerate(joint_labels) if i in np.r_[0:7, 9,
                                                                    12:14, 16]]
target_joints = ['lsho', 'lelb', 'lwri', 'rsho', 'relb',
                 'rwri', 'leye', 'reye', 'nose']
# indices of the needed joints in the coordinates array
joints_loc_id = np.r_[0:6, 12:14, 16]


def joint_coordinates(joint):
    """Store necessary coordinates to a list."""
    joint_coor = []
    # Take mean of the leye, reye, nose to obtain coordinates for the head
    joint['head'] = (joint['leye']+joint['reye']+joint['nose'])/3
    joint_coor.extend(joint['lwri'].tolist())
    joint_coor.extend(joint['lelb'].tolist())
    joint_coor.extend(joint['lsho'].tolist())
    joint_coor.extend(joint['head'].tolist())
    joint_coor.extend(joint['rsho'].tolist())
    joint_coor.extend(joint['relb'].tolist())
    joint_coor.extend(joint['rwri'].tolist())
    return joint_coor


# load the indices matlab file
train_indices = loadmat('FLIC-full/tr_plus_indices.mat')
# reshape the training_indices array
train_indices = train_indices['tr_plus_indices'].reshape(-1,)

# empty list to store train image ids
train_ids = []
# empty list to store train joints
train_jts = []
# empty list to store test image ids
test_ids = []
# empty list to store test joints
test_jts = []

for i, example in enumerate(examples):
    # image id
    file_name = example[3][0]
    # joint coordinates
    joint = example[2].T
    # dictionary that goes into the joint_coordinates function
    joints = dict(zip(target_joints,
                      [x for k, x in enumerate(joint) if k in joints_loc_id]))
    # obtain joints for the task
    joints = joint_coordinates(joints)
    # use train indices list to decide if an image is to be used for training
    # or testing
    if i in train_indices:
        train_ids.append(file_name)
        train_jts.append(joints)
    else:
        test_ids.append(file_name)
        test_jts.append(joints)

# Concatenate image ids dataframe and the joints dataframe and save it as a csv
train_df = pd.concat([pd.DataFrame(train_ids), pd.DataFrame(train_jts)],
                     axis=1)
test_df = pd.concat([pd.DataFrame(test_ids), pd.DataFrame(test_jts)], axis=1)
train_df.to_csv('FLIC-full/train_joints.csv', index=False, header=False)
test_df.to_csv('FLIC-full/test_joints.csv', index=False, header=False)

# load train_joints.csv
train_data = pd.read_csv('FLIC-full/train_joints.csv', header=None)
# load test_joints.csv
test_data = pd.read_csv('FLIC-full/test_joints.csv', header=None)

# train image ids
train_image_ids = train_data[0].values
# train joints
train_joints = train_data.iloc[:, 1:].values
# test image ids
test_image_ids = test_data[0].values
# test joints
test_joints = test_data.iloc[:, 1:].values

model_data(train_image_ids, train_joints, train=True)
model_data(test_image_ids, test_joints, train=False)

# Number of epochs
epochs = 3
# Batchsize
batch_size = 128
# Optimizer for the model
optimizer = Adam(lr=0.0001, beta_1=0.5)
# Shape of the input image
input_shape = (224, 224, 3)
# Batch interval at which loss is to be stores
store = 40

# load the vgg16 model
model = applications.VGG16(weights="imagenet", include_top=False,
                           input_shape=input_shape)

# set layers as non trainable
for layer in model.layers:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)

# Dense layer with 14 neurons for predicting 14 numeric values
predictions = Dense(14, activation="relu")(x)
# creating the final model
model_final = Model(inputs=model.input, outputs=predictions)
# compile the model
model_final.compile(loss="mean_squared_error", optimizer=optimizer)
# load the train data
train = pd.read_csv('FLIC-full/train/train_joints.csv', header=None)
# split train into train and validation
train_img_ids, val_img_ids, train_jts, val_jts = train_test_split(
        train.iloc[:, 0], train.iloc[:, 1:], test_size=0.2, random_state=42)

# load validation images
val_images = np.array(
    [cv.imread('FLIC-full/train/{}'.format(w)) for w in val_img_ids.values])

# convert validation images to dtype float
val_images = val_images.astype(float)


def training(model, image_ids, joints, val_images, val_jts,
             batch_size=128, epochs=2):
    """Train vgg16."""
    # empty train loss and validation loss list
    loss_lst = []
    val_loss_lst = []
    count = 0  # counter
    count_lst = []

    # create shuffled batches
    batches = np.arange(len(image_ids)//batch_size)
    data_idx = np.arange(len(image_ids))
    random.shuffle(data_idx)
    print('......Training......')
    for epoch in range(epochs):
        for batch in (batches):
            # batch of training image ids
            imgs = image_ids[data_idx[batch*batch_size:(batch+1)*batch_size:]]
            # corresponding joints for the above images
            jts = joints[data_idx[batch*batch_size:(batch+1)*batch_size:]]
            # load the training image batch
            batch_imgs = np.array(
                    [cv.imread('FLIC-full/train/{}'.format(x)) for x in imgs])
            # fit model on the batch
            loss = model.train_on_batch(batch_imgs.astype(float), jts)
            if batch % 40 == 0:
                # evaluate model on validation set
                val_loss = model.evaluate(val_images, val_jts, verbose=0,
                                          batch_size=batch_size)
                # store train and val loss
                loss_lst.append(loss)
                val_loss_lst.append(val_loss)
                print('Epoch:{}, End of batch:{}, loss:{:.2f},val_loss:{:.2f}\
                '.format(epoch+1, batch+1, loss, val_loss))

                count_lst.append(count)
            else:
                print('Epoch:{}, End of batch:{}, loss:{:.2f}\
                '.format(epoch+1, batch+1, loss))
            count += 1
    count_lst.append(count)
    loss_lst.append(loss)
    val_loss = model.evaluate(val_images, val_jts, verbose=0,
                              batch_size=batch_size)
    val_loss_lst.append(val_loss)
    print('Epoch:{}, End of batch:{}, VAL_LOSS:{:.2f}\
    '.format(epoch+1, batch+1, val_loss))
    return model, loss_lst, val_loss_lst, count_lst


m, loss_lst, val_loss_lst, count_lst = training(model_final,
                                                train_img_ids.values,
                                                train_jts.values,
                                                val_images,
                                                val_jts.values,
                                                epochs=epochs,
                                                batch_size=batch_size)

# plot the learning
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(count_lst, loss_lst, marker='D', label='training_loss')
plt.plot(count_lst, val_loss_lst, marker='o', label='validation_loss')
plt.xlabel('Batches')
plt.ylabel('Mean Squared Error')
plt.title('Plot of MSE over time')
plt.legend(loc='upper right')
plt.show()

# test and save results
test_loss = test(m)

# print test_loss
print('Test Loss:', test_loss)

image_list = glob.glob('FLIC-full/test_plot/*.jpg')[8:16]

plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, (i+1))
    img = cv.imread(image_list[i])
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    plt.title('Green-True/Red-Predicted Joints')

plt.tight_layout()
plt.show()
