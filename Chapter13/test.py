"""This module contains the function to test the vgg16 model performance."""
from plotting import *

import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def test(model, nrows=200, batch_size=128):
    """Test trained vgg16."""
    # load the train data
    test = pd.read_csv('FLIC-full/test/test_joints.csv', header=None,
                       nrows=nrows)
    test_img_ids = test.iloc[:, 0].values

    # load validation images
    test_images = np.array(
            [cv.imread('FLIC-full/test/{}'.format(x)) for x in test_img_ids])

    # convert validation images to dtype float
    test_images = test_images.astype(float)

    # joints
    test_joints = test.iloc[:, 1:].values

    # evaluate
    test_loss = model.evaluate(test_images, test_joints,
                               verbose=0, batch_size=batch_size)

    # predict
    predictions = model.predict(test_images, verbose=0, batch_size=batch_size)

    # folder to save the results
    if not os.path.exists(os.path.join(os.getcwd(), 'FLIC-full/test_plot')):
        os.mkdir('FLIC-full/test_plot')

    for i, (ids, image, joint, pred) in enumerate(zip(test_img_ids,
                                                      test_images,
                                                      test_joints,
                                                      predictions)):
        joints = joint.tolist()
        joints = list(zip(joints[0::2], joints[1::2]))

        # plot original joints
        image = plot_joints(image.astype(np.uint8), joints,
                            groundtruth=True, text_scale=0.5)

        pred = pred.astype(np.uint8).tolist()
        pred = list(zip(pred[0::2], pred[1::2]))

        # plot predicted joints
        image = plot_joints(image.astype(np.uint8), pred,
                            groundtruth=False, text_scale=0.5)

        # save resulting images with the same id
        plt.imsave('FLIC-full/test_plot/'+ids, image)
    return test_loss
