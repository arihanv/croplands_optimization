#!/usr/bin/env python
import csv
import numpy as np
from threading import Thread, current_thread
from subprocess32 import Popen, PIPE
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import rotate
from skimage import filters
import cv2
import time
import math
from sklearn.metrics import confusion_matrix



def objfunction(w):
    millis = int(round(time.time() * 1000))
    print('Started: ' + str(datetime.now()) + ' (' + str(millis) + ')')
    # Read original image and edge ground_truth
    image = plt.imread('./original.jpg').mean(axis=-1)
    ground_truth = plt.imread('./best_map.png').mean(axis=-1)

    # use dilation on ground truth with kernel of dimensions i x i
    i = 7
    kernel = np.ones((i, i), np.uint8)
    ground_truth = cv2.dilate(ground_truth, kernel, iterations=1)

    # If width is odd, make it even
    if(w%2 == 0):
        w+=1

    # Function to create the filter
    def createFilter_v2(w=None, k1=None, k2=None, o='vert'):
        if(w != None):
            k1 = k2 = math.ceil(w/2)
        if(k1 != k2):
            print(f"k1 has size of {k1} and k2 has size of {k2}. k1 and k2 must be equal")
            return None
        f = np.zeros((2*w+1, 2*w+1))
        for i in range(k1):
            f[:, i] = -1
        for i in reversed(range(k1)):
            i = i+1
            f[:, -i] = 1
        if(o=='horz'):
            f = np.transpose(f)
        return f

    x = createFilter_v2(w)

    # Apply the filter on the image
    edge_pred = convolve2d(image, x, mode='same')


    #Use otsu histogram thresholding to find an optimal threshold to binarize the edge prediction image
    val = filters.threshold_otsu(edge_pred)
    final_map = np.zeros_like(edge_pred)

    # Binarize the image using the optimal threshold found using otsu
    final_map[edge_pred > val] = 1

    # Compute BER
    cm = confusion_matrix(ground_truth.argmax(axis=-1).ravel(), final_map.argmax(axis=-1).ravel())
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    ## if statement to avoid division by zero error
    if tp + fn > 0 and tn + fp > 0:
        BER = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    else:
        BER = 1


    # Plot ground truth and final map
    fig,ax = plt.subplots(1,2)
    ax[1].cla()
    ax[0].cla()
    ax[0].imshow(ground_truth,'Greys_r')
    ax[0].set_title(f"Ground Truth (Dilated by {i} x {i})")
    ax[1].imshow(final_map,'Greys_r')
    ax[1].set_title(f"Final Map with kernel width of {w}")
    fig.suptitle(f"Ground Truth vs. Final Map\nBER: {BER}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show(block=False)
    plt.pause(1.5)

    plt.savefig(f'comparisons/{w}_{w}')

    end_millis = int(round(time.time() * 1000))
    print('Ended: ' + str(datetime.now()) + ' (' + str(end_millis) + ')')
    print(f"Time taken is {(end_millis - millis) / 1000.0000} secs")
    return BER

# Call function
for i in np.arange(1, 10, 2):
    print(f"BER is {objfunction(i)} for width of {i}")
    print()