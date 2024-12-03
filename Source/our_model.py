"""
    Adapted from: https://github.com/zhqiu/contrastive-learning-iSogCLR/blob/main/
"""
import timm
from transformers import DistilBertModel
from losses import CyCLIP_Loss, SogCLR_Loss, VICReg_Loss

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim as optim
from cosine_lr import CosineLRScheduler

class CLIP(nn.Module):
    def __init__(self,               
                 image_encoder = None,
                 text_encoder = None,
                 embed_dim = 256,
                 init_model = True,
                 world_size = 1,
                 ita_type = 'cyclip',
                 sogclr_gamma = 0.9,
                 rho_I = 0.1,
                 rho_T = 0.1,
                 eta_init = 0.001,
                 tau_init = 0.01,
                 eta_sched = None,
                 eta_exp_gamma = 0.8,
                 beta_u = 0.9,
                 temp = 0.01,
                 learnable_temp = False,
                 personalized_tau = False,
                 bsz = 128,
                 vicreg_sim_coeff = 25.0, 
                 vicreg_std_coeff = 25.0,
                 use_temp_net = True,
                 alpha = 1.0,
                 distributed=False,
                 ):
        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp

        if self.learnable_temp:
            self.image_temp = nn.Parameter(torch.ones(2900000) * self.temp)
            self.text_temp = nn.Parameter(torch.ones(2900000) * self.temp)
    
        # Encoders specified by project
        self.visual_encoder = timm.create_model(image_encoder, pretrained=True)
        self.text_encoder = DistilBertModel.from_pretrained(text_encoder, attn_implementation="sdpa")

        # Projection Layers
        self.vision_proj = nn.Linear(2048, embed_dim)   
        self.text_proj = nn.Linear(768, embed_dim)

        # String representation of loss function
        self.ita_type = ita_type

        # Instantiate loss function
        if self.ita_type == 'cyclip':
            self.loss_fn = CyCLIP_Loss(world_size=world_size, temperature=self.temp)
        elif self.ita_type == 'vicreg':
            self.loss_fn = VICReg_Loss(world_size=world_size, dim_size=embed_dim, sim_coeff=vicreg_sim_coeff, std_coeff=vicreg_std_coeff)
        elif self.ita_type == 'sogclr':
            self.loss_fn = SogCLR_Loss(world_size=world_size, gamma=sogclr_gamma, temperature=self.temp)
        else:
            raise NotImplementedError

    def forward(self, image, text, idx, text_idx, epoch):
        if self.learnable_temp:
            with torch.no_grad():
                self.image_temp.clamp_(0.001, 0.5)
                self.text_temp.clamp_(0.001, 0.5)

        # Get features of image
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1) 

        # Get features of caption
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
        text_embeds = self.text_proj(text_output.last_hidden_state[:,0,:])
        text_feat = F.normalize(text_embeds, dim=-1)                 

        if self.ita_type == 'cyclip':
            loss_ita = self.loss_fn(image_feat, text_feat)
        elif self.loss_fn == 'vicreg':
            loss_ita = self.loss_fn(image_embeds, text_embeds)
        elif self.loss_fn == 'sogclr':
            loss_ita, _, _ = self.loss_fn(image_feat, text_feat, idx, text_idx, epoch)
        else:
            raise NotImplementedError

        return loss_ita


class CLIP_Network():
    def __init__(self, model, optim, tokenizer, params, **kwargs):
        self.model = model # Should be an instance of the CLIP class
        self.model = self.model.cuda()
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=params.epochs,
            t_mul=1.0,
            lr_min=params.min_lr,
            decay_rate=params.decay_rate,
            warmup_lr_init=params.warmup_lr,
            warmup_t=params.warmup_steps,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=params.seed,
        )

def create_optimizer(params, model):
    opt_params = model.parameters()
    weight_decay = params.weight_decay

    if params.optimizer == "adam":
        optimizer = optim.Adam(opt_params)
    elif params.optimizer == "adamw":
        optimizer = optim.AdamW(opt_params)
    elif params.optimizer == "sgd":
        optimizer = optim.SGD(opt_params, momentum=params.momentum, nesterov=True)
    return optimizer
