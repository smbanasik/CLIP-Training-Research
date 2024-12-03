import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from data_process import generate_loaders
from our_model import CLIP_Model
from losses import SogCLR_Loss

def plot_save(train_log, test_log, title, ylabel, filename):
    plt.rcParams["figure.figsize"] = (9,5)
    x=np.arange(len(train_log))
    plt.figure()
    plt.plot(x, train_log, linestyle='-', label='Train Set', linewidth=3)
    plt.plot(x, test_log,  linestyle='-', label='Test Set', linewidth=3)
    plt.title(title,fontsize=25)
    plt.legend(fontsize=15)
    plt.grid()
    plt.ylabel(ylabel, fontsize=25)
    plt.xlabel('Epoch', fontsize=25)
    plt.savefig(filename)

def plot(train_log, test_log, title, ylabel):
    plt.rcParams["figure.figsize"] = (9,5)
    x=np.arange(len(train_log))
    plt.figure()
    plt.plot(x, train_log, linestyle='-', label='Train Set', linewidth=3)
    plt.plot(x, test_log,  linestyle='-', label='Test Set', linewidth=3)
    plt.title(title,fontsize=25)
    plt.legend(fontsize=15)
    plt.grid()
    plt.ylabel(ylabel, fontsize=25)
    plt.xlabel('Epoch', fontsize=25)
    plt.show()

class HyperParams():
    def __init__(self):
        self.epochs = 30
        self.learn_rates = [0.0005, 0.001, 0.002]
        self.lr_decay = 0.1
        self.lr_epoch = 30
        self.batch_size = 32
        self.weight_decay = 0
        self.gamma = 0.1
        self.step_size = 15

# Constructor for the AdamW optimizer
# Use this instead of clip_model.parameters to allow us to adjust lr/wd for each component
# The encoders are pretrained and should have a lower learning rate to prevent overshooting
def get_adamw_optimizer(model, lr=1e-4, weight_decay=1e-2):
    params = [
        {'params': model.image_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.text_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.image_proj.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.text_proj.parameters(), 'lr': lr, 'weight_decay': weight_decay},
    ]
    return AdamW(params)

def get_adam_optimizer(model, lr=1e-4):
    params = [
        {'params': model.image_encoder.parameters(), 'lr': lr},
        {'params': model.text_encoder.parameters(), 'lr': lr},
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
    step = StepLR(step_size=parameters.step_size, gamma=parameters.gamma)
    scores = clip_model.train(train_loader, coco_loader, imagenet_loader, sog_loss, adam_w, step, 5)
    print(f'scores: {scores}')
    
if __name__ == '__main__':
    main()