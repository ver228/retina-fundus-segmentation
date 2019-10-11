#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:51:05 2019

@author: avelinojaver
"""
from model import DiceCoeff

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import tqdm
import time
import torch
from pathlib import Path
import shutil

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = save_dir / 'model_best.pth.tar'
        shutil.copyfile(checkpoint_path, best_path)

def train_one_epoch(basename, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    get_dice_coeff = DiceCoeff() 
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    avg_loss = 0
    avg_dice = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        preds = model(images)
        loss = criterion(preds, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            dice_coeff = get_dice_coeff(preds, targets).item()
        
        avg_loss += loss
        avg_dice += dice_coeff
        
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    avg_loss /= len(data_loader)
    avg_dice /= len(data_loader)
     
    #save data into the logger
    logger.add_scalar('train_loss', avg_loss, epoch)
    logger.add_scalar('train_dice_coeff', avg_dice, epoch)
    
    return avg_dice


@torch.no_grad()
def evaluate_one_epoch(basename, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    get_dice_coeff = DiceCoeff() 
    
    model.eval()
    header = f'{basename} Test Epoch: [{epoch}]'
    
    avg_loss = 0
    avg_dice = 0
    model_time_avg = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        
        preds = model(images)
        
        model_time_avg += time.time() - model_time
        
        loss = criterion(preds, targets)
        dice_coeff = get_dice_coeff(preds, targets).item()
        
        avg_loss += loss
        avg_dice += dice_coeff
        
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    model_time_avg /= len(data_loader)
    avg_loss /= len(data_loader)
    avg_dice /= len(data_loader)
     
    #save data into the logger
    logger.add_scalar('test_loss', avg_loss, epoch)
    logger.add_scalar('test_dice_coeff', avg_dice, epoch)
    logger.add_scalar('model_time', model_time_avg, epoch)
    
    return avg_dice

def train_model(save_prefix,
        model,
        device,
        criterion,
        train_flow,
        test_flow,
        optimizer,
        log_dir,
        
        lr_scheduler = None,
        
        batch_size = 128,
        n_epochs = 2000,
        num_workers = 4
        ):
    
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers
                            )
    
    test_loader = DataLoader(test_flow, 
                            batch_size = 1, 
                            shuffle=False, 
                            num_workers=num_workers
                            )

    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_score = 0
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
            
        train_one_epoch(save_prefix, 
                         model, 
                         criterion,
                         optimizer, 
                         lr_scheduler, 
                         train_loader, 
                         device, 
                         epoch, 
                         logger
                         )
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        dice_coef = evaluate_one_epoch(save_prefix, 
                         model, 
                         criterion,
                         optimizer, 
                         lr_scheduler, 
                         test_loader, 
                         device, 
                         epoch, 
                         logger
                          )
        
        
        
        desc = 'epoch {} , dice_coef = {}'.format(epoch, dice_coef)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
        
        
        is_best = dice_coef > best_score
        if is_best:
            dice_coef = best_score  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        
    
        