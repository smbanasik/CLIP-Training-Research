import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from data_process import generate_loaders
from our_model import CLIP_Model
from losses import SogCLR_Loss


class HyperParamsAndArgs():
    def __init__(self, **kwargs):
        self.epochs = 30
        self.batch_size = 128
        self.seed = 615

        self.output_dir = "../output"

        self.learn_rates = [0.0005, 0.001, 0.002]
        self.learn_rates_pesg = [0.02, 0.05, 0.1]
        self.lr_decay = 0.1
        self.lr_epoch = 30
        self.weight_decay = 0
        self.gamma = 0.1
        self.step_size = 15

# Constructor for the AdamW optimizer
# Use this instead of clip_model.parameters to allow us to adjust lr/wd for each component
# The encoders are pretrained and should have a lower learning rate to prevent overshooting
def get_adamw_optimizer(model, lr=1e-4, weight_decay=1e-2):
    params = [
        {'params': model.image_encoder.parameters(), 'lr': lr/10, 'weight_decay': weight_decay},
        {'params': model.text_encoder.parameters(), 'lr': lr/10, 'weight_decay': weight_decay},
        {'params': model.image_proj.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.text_proj.parameters(), 'lr': lr, 'weight_decay': weight_decay},
    ]
    return AdamW(params)

def get_adam_optimizer(model, lr=1e-4):
    params = [
        {'params': model.image_encoder.parameters(), 'lr': lr/10},
        {'params': model.text_encoder.parameters(), 'lr': lr/10},
        {'params': model.image_proj.parameters(), 'lr': lr},
        {'params': model.text_proj.parameters(), 'lr': lr},
    ]
    return Adam(params)

def main():
    parameters = HyperParams()
    train_loader, coco_loader, imagenet_loader = generate_loaders(parameters)
    clip_model = CLIP_Model()
    # Loss functions 
    sog_loss = SogCLR_Loss(len(train_loader))
    # Optimizers
    adam_w = get_adamw_optimizer(clip_model, parameters.lr, parameters.weight_decay)
    adam = get_adam_optimizer(clip_model, parameters.lr)
    # Scheduler
    step = StepLR(optimizer=adam_w, step_size=parameters.step_size, gamma=parameters.gamma)
    scores = clip_model.train(train_loader, coco_loader, imagenet_loader, sog_loss, adam_w, step, 5)
    print(f'scores: {scores}')
    
if __name__ == '__main__':
    main()