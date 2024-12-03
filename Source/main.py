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

from data_process import generate_loaders

import pipeline as pipe
import our_model as our

def main(params):
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    train_loader, coco_loader, imagenet_loader = generate_loaders(params)

    print("Creating model")
    model = our.CLIP(image_encoder=params.image_encoder, text_encoder=params.text_encoder, embed_dim=args.embed_dim, init_model=args.True, bsz=params.batch_size,
                  world_size=1, ita_type=params.loss_type, sogclr_gamma=params.sogclr_gamma, rho_I=params.rho_I, rho_T=params.rho_T, tau_init=params.tau_init,
                  eta_init=params.eta_init, beta_u=params.beta_u, temp=params.temp, learnable_temp=params.learnable_temp,
                  vicreg_sim_coeff=params.vicreg_sim_coeff, vicreg_std_coeff=params.vicreg_std_coeff, personalized_tau=params.personalized_tau, 
                  use_temp_net=params.isogclr_temp_net, alpha=params.alpha, distributed=False)
    tokenizer = AutoTokenizer.from_pretrained(params.text_encoder)
    # TODO: make this work
    optimizer = create_optimizer(params, model)

    network = our.CLIP_Network(model, optimizer, tokenizer)
    
    
    print("--Begin training--")
    start_time = time.time()

    train_log = []
    test_log = []

    test_best = 0
    train_list_AUPRC, test_list_AUPRC = [], []
    train_list_AUROC, test_list_AUROC = [], []
    for epoch in range(params.epochs):
        
        if(params.is_training)
            pass # TODO: do train

        if(params.is_evaluating)
            pass # TODO: do eval

        # TODO: print results

        if(params.is_training)
            pass # TODO: torch save

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
        self.optimizer = "adamW"
        self.schedular = "cosine"

        self.is_evaluating = False
        self.is_training = True

        self.learn_rate = 2e-4
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

        self.learnable_temp = True
        self.personalized_tau = True
        self.isogclr_temp_net = True

if __name__ == '__main__':
    params = HyperParamsAndArgs()
    main(params)