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
        self.gammas = [0.2, 0.5, 0.8]
        self.decay_epochs = []

def main():

    parameters = HyperParamsAndArgs()
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # TODO: make sure this function does what its intended to do
    train_loader, coco_loader, imagenet_loader = generate_loaders(parameters)

    # TODO: create model here
    # Ensure model gets moved to CUDA.
    model = Model_Default(learn_rate, 0.003)
    '''
    model = CLIP(image_encoder=args.image_encoder, text_encoder=args.text_encoder, embed_dim=args.embed_dim, init_model=args.init_model, bsz=args.batch_size_train*args.world_size,
                  world_size=args.world_size, ita_type=args.ita_type, sogclr_gamma=args.sogclr_gamma, rho_I=args.rho_I, rho_T=args.rho_T, tau_init=args.tau_init,
                  eta_init=args.eta_init, beta_u=args.beta_u, temp=args.temp, learnable_temp=args.learnable_temp,
                  vicreg_sim_coeff=args.vicreg_sim_coeff, vicreg_std_coeff=args.vicreg_std_coeff, personalized_tau=args.personalized_tau, 
                  use_temp_net=args.isogclr_temp_net, alpha=args.alpha, distributed=args.distributed)
    '''
    
    print("--Begin training--")

    train_log = []
    test_log = []

    test_best = 0
    train_list_AUPRC, test_list_AUPRC = [], []
    train_list_AUROC, test_list_AUROC = [], []
    for epoch in range(parameters.epochs):
        if epoch in parameters.decay_epochs:
            model.opt.update_lr(decay_factor=10)
            pass
        
        # TODO: Train function from pipeline.py, adapt to our network dataloader model

        # TODO: evaluate function

        # TODO: print results

        train_loss = []
        model.network.train()
        for data, targets in trainloader:
            data, targets = data.cuda(), targets.cuda()
            preds = model.network(data)
            preds = torch.sigmoid(preds)
            loss = model.loss_func(preds, targets.float())

            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            train_loss.append(loss.item())
        
        model.network.eval()
        train_pred_list = []
        train_true_list = []
        for train_data, train_targets in evalloader:
            train_data = train_data.cuda()
            train_pred = model.network(train_data)
            train_pred_list.append(train_pred.cpu().detach().numpy())
            train_true_list.append(train_targets.numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        train_ap = auc_prc_score(train_true, train_pred)
        train_list_AUPRC.append(train_ap)
        train_auc = auc_roc_score(train_true, train_pred)
        train_list_AUROC.append(train_auc)
        train_loss = np.mean(train_loss)
    
        test_pred_list = []
        test_true_list = [] 
        for test_data, test_targets in testloader:
            test_data = test_data.cuda()
            test_pred = model.network(test_data)
            test_pred_list.append(test_pred.cpu().detach().numpy())
            test_true_list.append(test_targets.numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        val_ap = auc_prc_score(test_true, test_pred)
        test_list_AUPRC.append(val_ap)
        val_auc =  auc_roc_score(test_true, test_pred)
        test_list_AUROC.append(val_auc)
        model.network.train()
        if test_best < val_ap:
                test_best = val_ap

        print("epoch: %s, train_loss: %.4f, train_auc: %.4f, test_auc: %.4f, lr: %.4f"%(epoch, train_loss, train_auc, val_auc, model.opt.lr ))    
        train_log.append(train_auc) 
        test_log.append(val_auc)

    if(test_best > best_lr[0]):
        best_lr = [test_best, learn_rate]

    plot(train_list_AUPRC, test_list_AUPRC, "CrossEntropyLoss PneumoniaMNIST - AUPRC", "AUPRC", "crossent_auprc" + "_lr" + str(learn_rate) + ".png")
    plot(train_list_AUROC, test_list_AUROC, "CrossEntropyLoss PneumoniaMNIST - AUROC", "AUROC", "crossent_auroc" + "_lr" + str(learn_rate) + ".png")
    print("Best hyper parameters - LR:", best_lr[1])

if __name__ == '__main__':
    main()