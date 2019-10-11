#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:38:07 2019

@author: avelinojaver
"""
import cv2
from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm
import pickle

def preprocess_set(root_data_dir, set_type):
    mask_dir = root_data_dir / set_type /  '1st_manual'
    images_dir = root_data_dir / set_type /  'images'
    
    mask_files = {int(x.stem[:2]) : x for x in mask_dir.glob('*.gif')}
    images_files = {int(x.stem[:2]) : x for x in images_dir.glob('*.tif')}

    
    assert images_files.keys() == mask_files.keys()
    
    inds = sorted(list(images_files.keys()))
    
    set_data = []
    for ind in tqdm(inds, desc = f'Preprocessing {set_type}...'):
        img = cv2.imread(str(images_files[ind]))
        mask = io.imread(str(mask_files[ind]))
        mask = np.array(mask).astype(np.float32)/255
        
        #gray scaling
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        
        #image normalization
        bot, top = cl1.min(), cl1.max()
        img_norm = (cl1 - bot) / (top - bot)
        
        #gamma correction
        img_gamma = img_norm**1.2
        img_corrected = img_gamma.astype(np.float32)    
    
        set_data.append((img_corrected, mask))
        
    return set_data

def preprocess_DRIVE(root_data_dir, save_name = './DRIVE_preprocessed.p'):
    root_data_dir = Path(root_data_dir)
    save_name = Path(save_name)
    save_name.parent.mkdir(parents = True, exist_ok = True)
    
    
    processed_data = preprocess_DRIVE(root_data_dir)
    with open(save_name, 'wb') as fid:
        pickle.dump(processed_data, fid)
    
    processed_data = {}
    for set_type in ['training', 'test']:
        
        set_data = preprocess_set(root_data_dir, set_type)
            
        processed_data[set_type] = set_data

    return processed_data


if __name__ == '__main__':
    
    
    root_data_dir = Path.home() / 'Desktop/retina/DRIVE/'
    preprocess_DRIVE(root_data_dir, save_name = './DRIVE_preprocessed.p')
    
    
    