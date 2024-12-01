import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data_process import generate_loaders

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
        self.epochs = 15
        self.learn_rates = [0.0005, 0.001, 0.002]
        self.learn_rates_pesg = [0.02, 0.05, 0.1]
        self.lr_decay = 0.1
        self.lr_epoch = 30
        self.batch_size = 128
        self.weight_decay = 0
        self.gammas = [0.2, 0.5, 0.8]
        self.decay_epochs = []

def main():

    parameters = HyperParams()
    train_loader, coco_loader, imagenet_loader = generate_loaders(parameters)

    best_lr = [0, 0]
    for learn_rate in parameters.learn_rates:
        model = Model_Default(learn_rate, 0.003)

        print('Begin Training with Learn Rate', learn_rate)
        print('-'*30)

        train_log = []
        test_log = []

        test_best = 0
        train_list_AUPRC, test_list_AUPRC = [], []
        train_list_AUROC, test_list_AUROC = [], []
        for epoch in range(parameters.epochs):
            if epoch in parameters.decay_epochs:
                model.opt.update_lr(decay_factor=10)
                pass
            
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