#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:24:50 2019

@author: avelinojaver
"""
from transforms import RandomCrop, AffineTransform, RemovePadding, RandomVerticalFlip, RandomHorizontalFlip, Compose

import random
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import math
import numpy as np

class TrainingFlow(Dataset):
    def __init__(self, 
                 data, 
                 roi_size = 64,
                 zoom_range = (0.8, 1.25),
                 samples_per_epoch = 100000,
                 ):
        self.data = data
        self.roi_size = roi_size
        
        rotation_pad_size = math.ceil(self.roi_size*(math.sqrt(2)-1)/2)
        padded_roi_size = roi_size + 2*rotation_pad_size
        
        self.transforms = [
                RandomCrop(padded_roi_size),
                AffineTransform(zoom_range),
                RemovePadding(rotation_pad_size),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                
                ]
        
        self.transforms = Compose(self.transforms)
        
        self.samples_per_epoch = samples_per_epoch
    
    def __getitem__(self, ind):
        X, Y = random.choice(self.data)
        X, Y = self.transforms(X, Y)
        
        X, Y = np.ascontiguousarray(X), np.ascontiguousarray(Y)
        
        return X[None], Y[None] #add the extra channel for pytorch conv compatibility
    
    
    def __len__(self):
        return self.samples_per_epoch


class TestFlow(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, ind):
        X, Y = self.data[ind]
        
        return X[None], Y[None] #add the extra channel for pytorch conv compatibility
    
    
    def __len__(self):
        return len(self.data)


def load_flows(src_file = './DRIVE_preprocessed.p'):
    src_file = Path(src_file)
    with open(src_file, 'rb') as fid:
        data = pickle.load(fid)
    
    train_flow = TrainingFlow(data['training'])
    test_flow = TestFlow(data['test'])
    return train_flow, test_flow
    

if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    train_flow, test_flow = load_flows('./DRIVE_preprocessed.p')
    
    
    X, Y = train_flow[0]
    
    fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
    axs[0].imshow(X[0])
    axs[1].imshow(Y[0])
    
    X, Y = test_flow[0]
    
    fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
    axs[0].imshow(X[0])
    axs[1].imshow(Y[0])
    