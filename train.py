#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:17:15 2019

@author: avelinojaver
"""

from engine import train_model
from flow import load_flows
from datetime import datetime
from model import unet_constructor, DiceCoeff

import torch
from pathlib import Path
from torch import nn

LOG_DIR_DFLT = Path.home() / 'workspace/retina_fundus/results/'

def get_device(cuda_id):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device

def get_optimizer(optimizer_name, model, lr, weight_decay = 0.0, momentum = 0.9):
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr = lr, momentum = momentum, weight_decay = weight_decay)
    else:
        raise ValueError('Invalid optimizer name {}'.format(optimizer_name))
    return optimizer

def get_criterion(loss_name):
    if loss_name == 'BCE':
        return nn.BCELoss()
    elif loss_name == 'dice':
        return DiceCoeff()
    else:
        raise ValueError(f'Not implemented `{loss_name}`.')

def main(
    cuda_id = 0,
    loss_name = 'BCE',
    optimizer_name = 'adam',
    lr = 256e-5,
    log_dir = LOG_DIR_DFLT,
    batch_size = 256,
    n_epochs = 2000,
    num_workers = 0
    ):
    
    train_flow, test_flow = load_flows('./DRIVE_preprocessed.p')
    model = unet_constructor(1, 1)
    criterion = get_criterion(loss_name)
    
    optimizer = get_optimizer(optimizer_name, model, lr = 1e-5)
    device = get_device(cuda_id)
    
    
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_prefix = f'retina-fundus_unet_{loss_name}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}_lr{lr}_batch{batch_size}'
    
    train_model(save_prefix,
            model,
            device,
            criterion,
            train_flow,
            test_flow,
            optimizer,
            log_dir,
            
            lr_scheduler = None,
            
            batch_size = batch_size,
            n_epochs = n_epochs,
            num_workers = num_workers
            )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
