import timm 
import torch
import torch.cuda
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
from losses import SogCLR_Loss

class CLIP_Model():
    def __init__(self, latent_dim=256):
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.image_encoder = timm.create_model('resnet50', pretrained=True).to(self.device)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased", attn_implementation="sdpa")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        image_feature_dim = 1000 # ResNet50 output dimension
        text_feature_dim = 768   # DistilBERT output dimension

        # Projection Layers
        self.image_proj = nn.Linear(image_feature_dim, latent_dim).to(self.device)
        self.text_proj = nn.Linear(text_feature_dim, latent_dim).to(self.device)

    # Encode images/text in latent space
    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        image_features = self.image_proj(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)

        text_features = self.text_encoder(**texts).last_hidden_state[:, 0]
        text_features = self.text_proj(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        return image_features, text_features
    
    def train(self, train_loader, coco_loader, imagenet_loader, loss_fn, opt, sched, max_epochs=30, log_interval=10):

        self.image_proj.train()
        self.text_proj.train()

        img_to_text, text_to_image, top_1 = [], [], []
        need_indices = isinstance(loss_fn, SogCLR_Loss)
        
        print('-------------- Begin Training --------------')
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            self.image_encoder.train()
            self.text_encoder.train()
            for batch_idx, (images, captions, indices) in enumerate(train_loader):
                if batch_idx * train_loader.batch_size > 50:
                    break
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                opt.zero_grad()
                image_features, text_features = self.forward(images, captions)
                if need_indices:
                    loss = loss_fn(image_features, text_features, indices, indices)
                else:
                    loss = loss_fn(image_features, text_features)
                epoch_loss += loss.item()
                
                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                loss.backward()
                opt.step()

            sched.step()
               
            # Start validation
            self.image_encoder.eval()
            self.text_encoder.eval()

            # Evaluate metrics
            img_to_text_r1, text_to_img_r1 = self.evaluate_image_and_text(coco_loader)
            top1_accuracy = self.evaluate_top1_classification(imagenet_loader)

            # Print and store results
            print(f"Epoch {epoch + 1}/{max_epochs} - Loss: {epoch_loss:.4f} - "
                f"I_to_T R1: {img_to_text_r1:.4f} T_to_I R1: {text_to_img_r1:.4f} ZS acc: {top1_accuracy:.4f}")
            img_to_text.append(img_to_text_r1)
            text_to_image.append(text_to_img_r1)
            top_1.append(top1_accuracy)

        return img_to_text, text_to_image, top_1

    def evaluate_image_and_text(self, coco_loader):
        """
        Evaluate both Image-to-Text and Text-to-Image R@1 in a single pass through coco_loader.
        """
        image_features = []
        text_features = []
        for images, captions in coco_loader:
            # Move data to the correct device
            images = images.to(self.device)
            input_ids = captions['input_ids'].to(self.device)
            attention_mask = captions['attention_mask'].to(self.device)
            
            with torch.no_grad():
                # Extract and normalize image features
                img_feats = self.image_proj(self.image_encoder(images))
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                image_features.append(img_feats)
                
                # Extract and normalize text features
                text_rep = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
                txt_feats = self.text_proj(text_rep)
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

    def evaluate_top1_classification(self, imagenet_loader):
        """Evaluate Top-1 Classification Accuracy."""
        correct, total = 0, 0
        for images, labels in imagenet_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.image_encoder(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total

