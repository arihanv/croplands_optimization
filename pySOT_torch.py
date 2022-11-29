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
import time
import math
from sklearn.metrics import confusion_matrix

class TorchOptim:

    bestResult = 1000
    f_eval_count = 0
    seed = 139
    server = 'unset'

    # Hyperparameter to optimise:
    hyper_map = {
            'w' : 0, 
    }

    def __init__(self, seed, server, dim=1):

        self.seed = seed
        self.server = server
        self.f_eval_count = 0
        m = self.hyper_map 

        self.ground_truth = plt.imread()

        self.xlow = np.zeros(dim)
        self.xup = np.zeros(dim)

        # human value 0.0000
        self.xlow[m['w']] = 1
        self.xup[m['w']] =  20

    
        self.dim = dim
        self.info = 'Optimise a simple MLP network over MNIST dataset'
        # self.continuous = np.arange(0, 4)
        self.integer = np.arange(0, dim)

    def print_result_directly(self, x, result):
        self.f_eval_count = self.f_eval_count + 1
        experimentId = 'p-'+str(len(x))+'-'+str(self.f_eval_count)+'-'+self.seed+'-'+self.server
        fileId = 'p-'+str(len(x))+'-'+self.seed+'-'+self.server
        millis = int(round(time.time() * 1000))

        if self.bestResult > result:
            self.bestResult = result
        row = [self.bestResult, -1, result, -1, self.f_eval_count, millis] 
        for xi in range(0, len(x)):
            row.append(x[xi])
        with open('logs/'+fileId+'-output.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        


    def objfunction(self, w):
        
        # If the inputted width is even, make it odd

        image = plt.imread('./original.jpg').mean(axis=-1)
        ground_truth = plt.imread('./best_map.png').mean(axis=-1)
        # use dilation on ground truth

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
            print(k1)
            for i in range(k1):
                f[:, i] = -1
            for i in reversed(range(k1)):
                i = i+1
                f[:, -i] = 1
            if(o=='horz'):
                f = np.transpose(f)
            return f
        
        x = createFilter_v2(w)

        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

        edge_pred = convolve2d(image, x, mode='same')

        #Use otsu histogram thresholding to find an optimal threshold to binarize the edge prediction image
        val = filters.threshold_otsu(edge_pred)

        final_map = np.zeros_like(edge_pred)

        # Binarize the image using the optimal threshold found using otsu
        final_map[edge_pred > val] = 1

        cm = confusion_matrix(ground_truth.argmax(axis=-1).ravel(), final_map.ravel())
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        # Compute BER
        # if statement to avoid division by zero error
        if tp + fn > 0 and tn + fp > 0:
            BER = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        else:
            BER = 1
        
        print(BER)



        # put a threshold, to binarize it
        # otsu thresholding
        # histogram threshsholding
        # maximum distances from the peaks

        # edge greater than val is one, anything below dont change value


        # cm = confusion_matrix(new_GT.argmax(axis=-1).ravel(), final_pred.ravel())

        # make sure tp and fn exist, if not then return 1, min is 0

        # tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        # BER = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))

        # # after filter is obtained apply filter with convolution on the image, edge map, compare to ground truth
        # #compare that prediction to the ground truth, 

        # perform closing algorithm or smoothing on the edge map image

        # use morphological operations from opencv python

        # use dilation ******* using open cv2

        # manually create filter and find various metrics to see which one is best
        #BER metric (Balanced error rate) to measure performance
        # don't do 1 - BER,  minimize itself

        #Mean IOU metric (Amount of alignment of edges between ground truth and prediction)
        # return (1 - mean IOU)

        #assess precision, recall -> High f1 for success
        #1 - f1 score

        #compare final_map to ground truth

        # find boundary pixel accuracy metric

        # # return 1- accuracy

        

        self.f_eval_count = self.f_eval_count + 1
        experimentId = 'p-'+str(len(x))+'-'+str(self.f_eval_count)+'-'+self.seed+'-'+self.server
        fileId = 'p-'+str(len(x))+'-'+self.seed+'-'+self.server
        m = self.hyper_map


        exp_arg = []
        exp_arg.append('th'),
        exp_arg.append('eval_mnist_GPU.lua')
        exp_arg.append('--mean')
        exp_arg.append(str(x[m['mean']]))
        exp_arg.append('--std')
        exp_arg.append(str(x[m['std']]))
        exp_arg.append('--learnRate')
        exp_arg.append(str(x[m['learnRate']]))
        exp_arg.append('--momentum')
        exp_arg.append(str(x[m['momentum']]))
        exp_arg.append('--epochs')
        exp_arg.append(str(x[m['epochs']]))
        exp_arg.append('--hiddenNodes')
        exp_arg.append(str(x[m['hiddenNodes']]))
        exp_arg.append('--experimentId')
        exp_arg.append(experimentId)
        exp_arg.append('--seed')
        exp_arg.append(self.seed)
        
        millis_start = int(round(time.time() * 1000))
        proc = Popen(exp_arg, stdout=PIPE)
        out, err = proc.communicate()
        
        if proc.returncode == 0:
            results = out.split('###')
            result = float(results[0])
            testResult = float(results[1])
            millis = int(round(time.time() * 1000))
            f_eval_time = millis - millis_start

            if self.bestResult > result:
                self.bestResult = result
            
            row = [self.bestResult, f_eval_time, result, testResult, self.f_eval_count, millis] 
            for xi in range(0, len(x)):
                row.append(x[xi])
            with open('logs/'+fileId+'-output.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return result
        else:
            print(err)
            raise ValueError('Function evaluation error')
    
    def print_parameters(self, x):

        print(current_thread())
        m = self.hyper_map
        print('')
        for p in m:
             print(p+'\t : %g' % float(x[m[p]]))
