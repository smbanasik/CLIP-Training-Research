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

from models.model_clip import CLIP
from transformers import AutoTokenizer, RobertaTokenizer

import utils
import shutil
from dataset import create_train_dataset, create_val_dataset, create_sampler, create_train_loader, create_val_loader
from scheduler import create_scheduler
from optim import create_optimizer
from zeroshot_transfer.classes import CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES

from tqdm import tqdm
# optimizer, tokenizer, epoch, max_epoch, warmup_steps, device, scheduler, args
def train(network, data_loader, parameters, current_epoch):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_temp_net', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_image_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('avg_text_tau', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('cur_eta', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_image', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_tau_text', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_I', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('b_T', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('v', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('lamda', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_image_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('weights_text_pos', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(image, text, idx, text_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)   
        idx = idx.to(device, non_blocking=True)
        text_idx = text_idx.to(device, non_blocking=True)   
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
        
        # set learning rate for temperature network
        optimizer.param_groups[2]["lr"] = optimizer.param_groups[0]["lr"] / 10.0

        loss_ita, info_dict = model(image, text_input, idx=idx, text_idx=text_idx, epoch=epoch, max_epoch=max_epoch)
        loss_ita.backward()
        optimizer.step()
        
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
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and scheduler is not None: 
            scheduler.step(i//step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


"""
    zero-shot transfer
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/evaluate.py#L42
"""
def create_zeroshot_dataloader(dataset_name, data_folder, image_size):
    assert dataset_name in ['cifar10', 'cifar100', 'imagenet']

    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_folder, download=False, train=False, transform=val_transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_folder, download=False, train=False, transform=val_transform)
    else:
        dataset = datasets.ImageFolder(root=data_folder, transform=val_transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                              num_workers=2, pin_memory=True)

    data_loader.num_samples = len(dataset)

    return data_loader



@torch.no_grad()
def zeroshot_transfer(model, data_loader, dataset_name, tokenizer, device):
    model.eval()

    if dataset_name == 'cifar10':
        config = CIFAR10_CLASSES
    elif dataset_name == 'cifar100':
        config = CIFAR100_CLASSES
    elif dataset_name == 'imagenet':
        config = IMAGENET_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    classes, templates = config["classes"], config["templates"]

    text_embeddings = []
    for c in classes:
        texts = [template(c) for template in templates]
        text_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_outputs = model.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask, output_hidden_states=False)  
        text_embeds = F.normalize(model.text_proj(text_outputs.last_hidden_state[:,0,:]), dim=-1)
        text_embed = text_embeds.mean(dim=0)
        text_embed /= text_embed.norm()
        text_embeddings.append(text_embed)

    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    topk = [1, 3, 5, 10]
    correct = {k: 0 for k in topk}

    for image, label in data_loader:
        image, label = image.to(device), label.to(device)
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat)            
        image_embedding = F.normalize(image_embed, dim=-1)

        logits = image_embedding @ text_embeddings
        ranks = logits.topk(max(topk), 1)[1].T
        predictions = ranks == label

        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    results = {f"zeroshot_top{k}": 100.0 * correct[k] / data_loader.num_samples for k in topk}

    return results



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