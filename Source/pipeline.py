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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from our_model import CLIP
from transformers import AutoTokenizer, RobertaTokenizer

import utils
import shutil

from tqdm import tqdm
def train(network, data_loader, parameters, current_epoch):
    # train
    network.model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Train Epoch: [{}]'.format(current_epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = parameters.warmup_steps*step_size  

    for i,(image, text, idx, text_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        network.optimizer.zero_grad()

        image = image.cuda()   
        idx = idx.cuda()
        text_idx = text_idx.cuda() 
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").cuda()
        
        # set learning rate for temperature network
        network.optimizer.param_groups[2]["lr"] = optimizer.param_groups[0]["lr"] / 10.0

        loss_ita, info_dict = network.model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
        loss_ita.backward()
        network.optimizer.step()
        
        metric_logger.update(loss_ita=loss_ita.item())

        if args.ita_type in ['sogclr_dro', 'isogclr_new']:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)
        elif args.ita_type == 'isogclr_new_v2':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=info_dict['cur_eta'])
            metric_logger.update(grad_tau_image=info_dict['grad_tau_image'])
            metric_logger.update(grad_tau_text=info_dict['grad_tau_text'])
            metric_logger.update(b_I=info_dict['b_I'])
            metric_logger.update(b_T=info_dict['b_T'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(v=info_dict['v'])
            metric_logger.update(lamda=info_dict['lamda'])
        elif args.ita_type == 'sogclr':
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=info_dict['lamda'])
        else:
            metric_logger.update(avg_image_tau=info_dict['avg_image_tau'])
            metric_logger.update(avg_text_tau=info_dict['avg_text_tau'])
            metric_logger.update(cur_eta=0.0)
            metric_logger.update(grad_tau_image=0.0)
            metric_logger.update(grad_tau_text=0.0)
            metric_logger.update(weights_image_pos=0.0)
            metric_logger.update(weights_text_pos=0.0)
            metric_logger.update(b_I=0.0)
            metric_logger.update(b_T=0.0)
            metric_logger.update(v=0.0)
            metric_logger.update(lamda=0.0)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_temp_net=optimizer.param_groups[2]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and network.scheduler is not None: 
            network.scheduler.step(i//step_size)

    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

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
def evaluation(model, data_loader, tokenizer, device, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, output_hidden_states=False)  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds,dim=0)
    
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embed = F.normalize(image_embed, dim=-1)      
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds.to(device) @ text_embeds.to(device).t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_i2t[start+i, topk_idx] = topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(sims_matrix[start:end]): 
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_t2i[start+i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
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