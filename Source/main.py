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
from our_model import CLIP, CLIP_Network, create_optimizer
from losses import SogCLR_Loss
from transformers import AutoTokenizer

import pipeline as pipe
import our_model as our

def main(params):
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    train_loader, coco_loader, imagenet_loader = generate_loaders(params)

    print("Creating model")
    model = our.CLIP(image_encoder=params.image_encoder, text_encoder=params.text_encoder, embed_dim=params.embed_dim, init_model=True, bsz=params.batch_size,
                  world_size=1, ita_type=params.loss_type, sogclr_gamma=params.sogclr_gamma, rho_I=params.rho_I, rho_T=params.rho_T, tau_init=params.tau_init,
                  eta_init=params.eta_init, beta_u=params.beta_u, temp=params.temp, learnable_temp=params.learnable_temp,
                  vicreg_sim_coeff=params.vicreg_sim_coeff, vicreg_std_coeff=params.vicreg_std_coeff, personalized_tau=params.personalized_tau, 
                  use_temp_net=params.isogclr_temp_net, alpha=params.alpha, distributed=False)
    tokenizer = AutoTokenizer.from_pretrained(params.text_encoder)
    optimizer = create_optimizer(params, model)

    network = our.CLIP_Network(model, optimizer, tokenizer, params)
    
    print("--Begin training--")
    start_time = time.time()

    for epoch in range(params.epochs):
        
        if(params.is_training):
            train_stats = pipe.train(network, train_loader, params, epoch)

        if(params.is_evaluating):
            pass # TODO: do eval

        if(params.is_training):
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},             
                             'epoch': epoch,
                             'data': 'coco',
                            }
            with open(os.path.join(params.output_dir, "coco_log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_obj = {
                    'model': model_without_ddp.state_dict()
                }
            torch.save(save_obj, os.path.join(params.output_dir, 'checkpoint_'+str(epoch+1)+'.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

class HyperParamsAndArgs():
    def __init__(self, **kwargs):
        # These cannot be changed - project requirements
        self.epochs = 30
        self.batch_size = 128
        self.text_encoder = "distilbert-base-uncased"
        self.image_encoder = "resnet50"
        # This shouldn't be changed
        self.seed = 615
        self.output_dir = "../output"
        # Anything below may be modified
        
        self.loss_type = "sogclr"
        self.optimizer = "adamw"
        self.schedular = "cosine"

        self.is_evaluating = False
        self.is_training = True
        self.warmup_steps = 2
        
        self.learn_rate = 2e-4
        self.weight_decay = 0.02
        self.momentum = 0.09
        self.sogclr_gamma = 0.8
        self.rho_I = 8.0
        self.rho_T = 8.0
        self.tau_init = 0.01
        self.eta_init = 0.001
        self.beta_u = 0.9
        self.temp = 0.01
        self.vicreg_sim_coeff = 25.0
        self.vicreg_std_coeff = 25.0
        self.alpha = 1.0
        self.embed_dim = 256
        self.decay_rate = 1
        self.warmup_lr = 1e-5
        self.min_lr = 1e-6

        self.learnable_temp = True
        self.personalized_tau = True
        self.isogclr_temp_net = True

# Removed optimizer constructor in place of my own. Once we confirm it works we can reimplement. We can find it in previous commits

if __name__ == '__main__':
    params = HyperParamsAndArgs()
    main(params)