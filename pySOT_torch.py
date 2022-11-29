#!/usr/bin/env python
import csv
import numpy as np
from threading import Thread, current_thread
from subprocess32 import Popen, PIPE
from datetime import datetime
import matplotlib.pyplot as plt
import time
import math

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
