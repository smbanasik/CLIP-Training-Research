"""
    Adapted from: https://github.com/zhqiu/contrastive-learning-iSogCLR/blob/main/
"""
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

from our_model import CLIP

# Train a CLIP_Network for one epoch
def train(network, train_loader, parameters, current_epoch, max_epoch=30):
    # Set to training mode
    network.model.train()
    epoch_loss = 0
    for images, cap_tokens, indices in enumerate(train_loader):
        
        network.optimizer.zero_grad()

        images = images.cuda()   
        indices = indices.cuda()
        cap_tokens = cap_tokens.cuda() 
        
        # set learning rate for temperature network to be 1/10 of the host network's lr
        network.optimizer.param_groups[2]["lr"] = network.optimizer.param_groups[0]["lr"] / 10.0

        loss_ita = network.model(images, cap_tokens, idx=indices, text_idx=indices, epoch=current_epoch, max_epoch=max_epoch)
        epoch_loss += loss_ita.item()
        loss_ita.backward()
        network.optimizer.step()
        
    if network.scheduler is not None and current_epoch % parameters.step_size == 0: 
        network.scheduler.step()

    return epoch_loss


# Top 1 Accuracy on the ImageNet validation set
@torch.no_grad()
def evaluate_top1_classification(model, imagenet_loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in imagenet_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model.visual_encoder(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

@torch.no_grad()
def evaluate_image_and_text(model, coco_loader, device):
    """
    Evaluate both Image-to-Text and Text-to-Image R@1 in a single pass through coco_loader.
    """
    image_features = []
    text_features = []
    for images, captions in coco_loader:
        # Move data to the correct device
        images = images.to(device)
        input_ids = captions['input_ids'].to(device)
        attention_mask = captions['attention_mask'].to(device)
        
        with torch.no_grad():
            # Extract and normalize image features
            img_feats = model.image_proj(model.visual_encoder(images))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            image_features.append(img_feats)
            
            # Extract and normalize text features
            text_rep = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            txt_feats = model.text_proj(text_rep)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            text_features.append(txt_feats)
    
    # Concatenate features
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    # Compute similarity matrices
    similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)

    # Image-to-Text R@1
    img_to_text_top_indices = similarity_matrix.argmax(dim=1)
    img_to_text_correct = sum(idx == i for i, idx in enumerate(img_to_text_top_indices))
    img_to_text_r1 = img_to_text_correct / len(image_features)

    # Text-to-Image R@1
    text_to_img_top_indices = similarity_matrix.argmax(dim=0)
    text_to_img_correct = sum(idx == i for i, idx in enumerate(text_to_img_top_indices))
    text_to_img_r1 = text_to_img_correct / len(text_features)

    return img_to_text_r1, text_to_img_r1



            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result