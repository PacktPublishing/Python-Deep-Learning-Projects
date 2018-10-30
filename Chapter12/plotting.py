"""This module contains functions to plot the joints and the limbs."""
import cv2 as cv
import numpy as np


def plot_limb(img, joints, i, j, color):
    """Limb plot function."""
    cv.line(img, joints[i], joints[j], (255, 255, 255), thickness=2,
            lineType=16)
    cv.line(img, joints[i], joints[j], color, thickness=1, lineType=16)
    return img


def plot_joints(img, joints, groundtruth=True, text_scale=0.5):
    """Joint and Limb plot function."""
    h, w, c = img.shape
    if groundtruth:
        # left hand to left elbow
        img = plot_limb(img, joints, 0, 1, (0, 255, 0))

        # left elbow to left shoulder
        img = plot_limb(img, joints, 1, 2, (0, 255, 0))

        # left shoulder to right shoulder
        img = plot_limb(img, joints, 2, 4, (0, 255, 0))

        # right shoulder to right elbow
        img = plot_limb(img, joints, 4, 5, (0, 255, 0))

        # right elbow to right hand
        img = plot_limb(img, joints, 5, 6, (0, 255, 0))

        # neck coordinate
        neck = tuple((np.array(joints[2]) + np.array(joints[4])) // 2)
        joints.append(neck)
        # neck to head
        img = plot_limb(img, joints, 3, 7, (0, 255, 0))
        joints.pop()

    # joints
    for j, joint in enumerate(joints):
        # plot joints
        cv.circle(img, joint, 5, (0, 255, 0), -1)
        # plot joint number black
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (0, 0, 0), thickness=2, lineType=16)
        # plot joint number white
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (255, 255, 255), thickness=1, lineType=16)

    else:
        # left hand to left elbow
        img = plot_limb(img, joints, 0, 1, (0, 0, 255))

        # left elbow to left shoulder
        img = plot_limb(img, joints, 1, 2, (0, 0, 255))

        # left shoulder to right shoulder
        img = plot_limb(img, joints, 2, 4, (0, 0, 255))

        # right shoulder to right elbow
        img = plot_limb(img, joints, 4, 5, (0, 0, 255))

        # right elbow to right hand
        img = plot_limb(img, joints, 5, 6, (0, 0, 255))

        # neck coordinate
        neck = tuple((np.array(joints[2]) + np.array(joints[4])) // 2)
        joints.append(neck)

        # neck to head
        img = plot_limb(img, joints, 3, 7, (0, 0, 255))
        joints.pop()

    # joints
    for j, joint in enumerate(joints):
        # plot joints
        cv.circle(img, joint, 5, (0, 0, 255), -1)
        # plot joint number black
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (0, 0, 0), thickness=3, lineType=16)
        # plot joint number white
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (255, 255, 255), thickness=1, lineType=16)

    return img
