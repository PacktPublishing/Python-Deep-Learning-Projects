"""This module contains functions to crop and resize images."""

import os
import cv2 as cv
import numpy as np
import pandas as pd


def image_cropping(image_id, joints, crop_pad_inf=1.4, crop_pad_sup=1.6,
                   shift=5, min_dim=100):
    """Crop Function."""
    # # image cropping
    # load the image
    image = cv.imread('FLIC-full/images/%s' % (image_id))
    # convert joint list to array
    joints = np.asarray([int(float(p)) for p in joints])
    # reshape joints to shape (7*2)
    joints = joints.reshape((len(joints) // 2, 2))
    # transform joints to list of (x,y) tuples
    posi_joints = [(j[0], j[1]) for j in joints if j[0] > 0 and j[1] > 0]
    # obtain the bounding rectangle using opencv boundingRect
    x_loc, y_loc, width, height = cv.boundingRect(np.asarray([posi_joints]))
    if width < min_dim:
        width = min_dim
    if height < min_dim:
        height = min_dim

    # # bounding rect extending
    inf, sup = crop_pad_inf, crop_pad_sup
    r = sup - inf
    # define width padding
    pad_w_r = np.random.rand() * r + inf  # inf~sup
    # define height padding
    pad_h_r = np.random.rand() * r + inf  # inf~sup
    # adjust x, y, w and h by the defined padding
    x_loc -= (width * pad_w_r - width) / 2
    y_loc -= (height * pad_h_r - height) / 2
    width *= pad_w_r
    height *= pad_h_r

    # # shifting
    x_loc += np.random.rand() * shift * 2 - shift
    y_loc += np.random.rand() * shift * 2 - shift

    # # clipping
    x_loc, y_loc, width, height = [int(z) for z in [x_loc, y_loc,
                                                    width, height]]
    x_loc = np.clip(x_loc, 0, image.shape[1] - 1)
    y_loc = np.clip(y_loc, 0, image.shape[0] - 1)
    width = np.clip(width, 1, image.shape[1] - (x_loc + 1))
    height = np.clip(height, 1, image.shape[0] - (y_loc + 1))
    image = image[y_loc: y_loc + height, x_loc: x_loc + width]

    # # joint shifting
    # adjust joint coordinates onto the padded image
    joints = np.asarray([(j[0] - x_loc, j[1] - y_loc) for j in joints])
    joints = joints.flatten()

    return image, joints


def image_resize(image, joints, new_size=224):
    """Resize Function."""
    orig_h, orig_w = image.shape[:2]
    joints[0::2] = joints[0::2] / float(orig_w) * new_size
    joints[1::2] = joints[1::2] / float(orig_h) * new_size
    image = cv.resize(image, (new_size, new_size),
                      interpolation=cv.INTER_NEAREST)
    return image, joints


def model_data(image_ids, joints, train=True):
    """Function to generate train and test data."""
    if train:
        # empty list
        train_img_joints = []
        # create train directory inside FLIC-full
        if not os.path.exists(os.path.join(os.getcwd(), 'FLIC-full/train')):
            os.mkdir('FLIC-full/train')

        for i, (image, joint) in enumerate(zip(image_ids, joints)):
            # crop the image using the joint coordinates
            image, joint = image_cropping(image, joint)
            # resize the cropped image to shape (224*224*3)
            image, joint = image_resize(image, joint)
            # save the image in train folder
            cv.imwrite('FLIC-full/train/train{}.jpg'.format(i), image)
            # store joints and image id/file name of the saved image in
            # the initialized list
            train_img_joints.append(['train{}.jpg'.format(i)] + joint.tolist())

        # convert to a dataframe and save as a csv
        pd.DataFrame(train_img_joints
                     ).to_csv('FLIC-full/train/train_joints.csv',
                              index=False, header=False)
    else:
        # empty list
        test_img_joints = []
        # create test directory inside FLIC-full
        if not os.path.exists(os.path.join(os.getcwd(), 'FLIC-full/test')):
            os.mkdir('FLIC-full/test')

        for i, (image, joint) in enumerate(zip(image_ids, joints)):
            # crop the image using the joint coordinates
            image, joint = image_cropping(image, joint)
            # resize the cropped image to shape (224*224*3)
            image, joint = image_resize(image, joint)
            # save the image in test folder
            cv.imwrite('FLIC-full/test/test{}.jpg'.format(i), image)
            # store joints and image id/file name of the saved image
            # in the initialized list
            test_img_joints.append(['test{}.jpg'.format(i)] + joint.tolist())

        # convert to a dataframe and save as a csv
        pd.DataFrame(test_img_joints).to_csv('FLIC-full/test/test_joints.csv',
                                             index=False, header=False)
