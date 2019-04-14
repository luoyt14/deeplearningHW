from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        err = input - target
        return np.sum(np.sum(err*err, 1)) / (2 * input.shape[0])

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        n = input.shape[0]
        x = input - np.max(input)
        # dem = np.sum(np.exp(x),1)
        # h = np.zeros(x.shape)
        # for i in range(n):
            # h[i] = np.exp(x[i]) / dem[i]
        h = np.exp(x) / (np.sum(np.exp(x),1).reshape(n,1))
        return np.sum(-np.sum(target * np.log(h+1e-3), 1)) / n

    def backward(self, input, target):
        '''Your codes here'''
        n = input.shape[0]
        x = input - np.max(input)
        # dem = np.sum(np.exp(x),1)
        # h = np.zeros(x.shape)
        # for i in range(n):
            # h[i] = np.exp(x[i]) / dem[i]
        h = np.exp(x) / (np.sum(np.exp(x),1).reshape(n,1))
        return - (target - h) / n