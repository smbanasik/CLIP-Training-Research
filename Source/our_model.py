import timm 
import torch
import torch.cuda
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F

class CLIP_Model():
    def __init__(self, latent_dim=256):
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.image_encoder = timm.create_model('resnet50', pretrained=True).to(self.device)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")

        image_feature_dim = 2048 # ResNet50 output dimension
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
        print('-------------- Begin Training --------------')
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            self.image_encoder.train()
            self.text_encoder.train()
            for batch_idx, (images, captions) in enumerate(train_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                opt.zero_grad()
                image_features, text_features = self.forward(images, captions)
                
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

            # Prepare validation features
            coco_images, coco_captions = next(iter(coco_loader))
            coco_images = coco_images.to(self.device)
            coco_image_features = self.image_proj(self.image_encoder(coco_images))
            coco_text_features = self.text_proj(self.text_encoder(coco_captions))

            # Normalize features
            coco_image_features = coco_image_features / coco_image_features.norm(dim=-1, keepdim=True)
            coco_text_features = coco_text_features / coco_text_features.norm(dim=-1, keepdim=True)

            # Evaluate on all metrics
            img_to_text_r1 = self.evaluate_image_to_text(coco_loader, coco_text_features)
            text_to_img_r1 = self.evaluate_text_to_image(coco_loader, coco_image_features)
            top1_accuracy = self.evaluate_top1_classification(imagenet_loader)

            print(f"Epoch {epoch + 1}/{max_epochs} - Loss: {epoch_loss:.4f} - I_to_T R1: {img_to_text_r1:.4f} T_to_I R1: {text_to_img_r1:.4f} ZS acc: {top1_accuracy:.4f}")
            img_to_text.append(img_to_text_r1)
            text_to_image.append(text_to_img_r1)
            top_1.append(top1_accuracy)
        
        return img_to_text, text_to_image, top_1
    
    def evaluate_image_to_text(self, image_loader, text_features):
        """Evaluate Text-to-Image R@1."""
        image_features = []
        for images, _ in image_loader:
            images = images.to(self.device)
            img_feats = self.image_proj(self.image_encoder(images))
            image_features.append(img_feats / img_feats.norm(dim=-1, keepdim=True))
        image_features = torch.cat(image_features, dim=0)

        similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)
        top_indices = similarity_matrix.argmax(dim=1)
        correct = sum(idx == i for i, idx in enumerate(top_indices))
        return correct / len(image_features)

    def evaluate_text_to_image(self, text_loader, image_features):
        """Evaluate Text-to-Image R@1."""
        text_features = []
        for _, captions in text_loader:
            captions = captions.to(self.device)
            text_feats = self.text_proj(self.text_encoder(captions))
            text_features.append(text_feats / text_feats.norm(dim=-1, keepdim=True))
        text_features = torch.cat(text_features, dim=0)

        similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)
        top_indices = similarity_matrix.argmax(dim=0)
        correct = sum(idx == i for i, idx in enumerate(top_indices))
        return correct / len(text_features)

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

