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

    # Get vertical and horizontal sobels
    vert_sobel = createFilter_v2(w)
    horz_sobel = createFilter_v2(w, o='horz')

    # Apply the filter on the image
    vert_edge_pred = convolve2d(image, vert_sobel, mode='same')
    horz_edge_pred = convolve2d(image, horz_sobel, mode='same')

    edge_pred = np.add(vert_edge_pred, horz_edge_pred)

    #Use otsu histogram thresholding to find an optimal threshold to binarize the edge prediction image
    # vert_val = filters.threshold_otsu(vert_edge_pred)
    # vert_final_map = np.zeros_like(vert_edge_pred)

    # horz_val = filters.threshold_otsu(horz_edge_pred)
    # horz_final_map = np.zeros_like(horz_edge_pred)

    val = filters.threshold_otsu(edge_pred)

    final_map = np.zeros_like(edge_pred)

    print(final_map.shape)
    print(ground_truth.shape)

    # Binarize the image using the optimal threshold found using otsu
    # vert_final_map[vert_edge_pred > vert_val] = 1
    # horz_final_map[horz_edge_pred > horz_val] = 1
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
    ax[1].set_title(f"Final Map with zeros width of {w}")
    fig.suptitle(f"Ground Truth vs. Final Map\nBER: {BER}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show(block=False)
    plt.pause(1.5)

    plt.savefig(f'comparisons/{w}_{w}')

    plt.close()

    end_millis = int(round(time.time() * 1000))
    print('Ended: ' + str(datetime.now()) + ' (' + str(end_millis) + ')')
    print(f"Time taken is {(end_millis - millis) / 1000.0000} secs")
    return BER

# Call function
for i in np.arange(1, 25, 2):
    print(f"BER is {objfunction(i)} for width of {i}")
    print()