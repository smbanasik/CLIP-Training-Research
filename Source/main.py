import numpy as np
import random
import time
import datetime
import os 
import torch
from transformers import DistilBertTokenizer

from data_process import generate_loaders
from our_model import create_optimizer

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
    optimizer = create_optimizer(params, model)
    tokenizer = DistilBertTokenizer.from_pretrained(params.text_encoder)
    network = our.CLIP_Network(model, optimizer, tokenizer, params)
    
    print("--Begin training--")
    start_time = time.time()

    # Train CLIP_Network model
    for epoch in range(params.epochs):
        epoch_loss = pipe.train(network, train_loader, params, epoch)
        if epoch % params.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': network.model.state_dict(),
                'optimizer_state_dict': network.optimizer.state_dict(),
                'scheduler_state_dict': network.scheduler.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, os.path.join(params.output_dir, 'checkpoint_'+str(epoch+1)+'.pth'))
        print(f'Epoch {epoch+1}/{params.epochs}- Loss: {epoch_loss:.4f}')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Start validation
    print('--Begin validation--')
    start_time = time.time()

    img_to_text_r1, text_to_img_r1 = pipe.evaluate_image_and_text(model, coco_loader, params.device)
    zero_shot_acc = pipe.evaluate_top1_classification(model, imagenet_loader, params.device)
    final_metric = (img_to_text_r1 + text_to_img_r1 + zero_shot_acc) / 3
    print(f'Results: Img-Text R1: {img_to_text_r1:.4f}, Text-Img R1: {text_to_img_r1:.4f}, Zero-Shot:{zero_shot_acc:.4f}')
    print(f'Final Metric: {final_metric:.4f}')
    return final_metric

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
        self.scheduler = "cosine"

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
        self.step_size = 15
        self.save_interval = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learnable_temp = True
        self.personalized_tau = True
        self.isogclr_temp_net = True

# Removed optimizer constructor in place of my own. Once we confirm it works we can reimplement. We can find it in previous commits

if __name__ == '__main__':
    params = HyperParamsAndArgs()
    main(params)