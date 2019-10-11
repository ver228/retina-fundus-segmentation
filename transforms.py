#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:40:53 2019

@author: avelinojaver
"""

import random
import numpy as np
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomCrop(object):
    def __init__(self, padded_roi_size = 512):
        self.padded_roi_size = padded_roi_size

    
    def __call__(self, image, target):
        
        #### select the limits allowed for a random crop
        xlims = (0, image.shape[1] - self.padded_roi_size - 1)
        ylims = (0, image.shape[0] - self.padded_roi_size - 1)
        
            
        #### crop with padding in order to keep a valid rotation 
        xl = random.randint(*xlims)
        yl = random.randint(*ylims)
        
        yr = yl + self.padded_roi_size
        xr = xl + self.padded_roi_size
    
        image_crop = image[yl:yr, xl:xr].copy()
        target_crop = target[yl:yr, xl:xr].copy()
        
        return image_crop, target_crop


class AffineTransform():
    def __init__(self, zoom_range = (0.9, 1.1), rotation_range = (-90, 90)):
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
    
    def __call__(self, image, target):
        theta = np.random.uniform(*self.rotation_range)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        
        cols, rows = image.shape[0], image.shape[1]
    
        M = cv2.getRotationMatrix2D((rows/2,cols/2), theta, scaling)
        
        offset_x = 0.
        offset_y = 0.
        translation_matrix = np.array([[1, 0, offset_x],
                                       [0, 1, offset_y],
                                       [0, 0, 1]])
        
        M = np.dot(M, translation_matrix)
        
        image_rot = cv2.warpAffine(image, M, (rows, cols))
        target_rot = cv2.warpAffine(target, M, (rows, cols))
        
        return image_rot, target_rot

class RemovePadding():
    def __init__(self, pad_size):
        self.pad_size = pad_size
    
    def __call__(self, image, target):
        ##### remove padding
        img_out = image[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        target_out = target[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        
        return img_out, target_out
    
class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[::-1]
            target = target[::-1]
            
        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, ::-1]
            target = target[:, ::-1]
            
        return image, target
