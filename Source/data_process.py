from torch.utils.data import Dataset, DataLoader
from PIL import Image

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torchvision import datasets
from transformers import DistilBertTokenizer

import json 
import os 
import torch

# CC3M Training Dataset
class CC3MDataset(Dataset):
    def __init__(self, captions_file, img_root_dir, transform=None):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            root_dir (str): Path to the root directory of the images.
        """

        self.img_root_dir = img_root_dir

        if transform:
            self.transform = transform 
        else:
            # Use timm's transformations instead of torch.compose, since they're targeted to ResNet50
            config = resolve_data_config({}, model='resnet50')
            self.transform  = create_transform(**config)        
        
        # Load captions and image paths from the JSON file
        with open(captions_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_root_dir, sample['image'])
        image = Image.open(img_path).convert('RGB')
        
        caption = sample['caption']
        # Return the image index, required by SogCLR loss function
        return image, caption, idx

# MS-COCO Validation Dataset for recall
class MSCOCODataset(Dataset):
    def __init__(self, captions_file, img_root_dir, transform=None):
        """
        Args:
            captions_file (str): Path to the captions JSON file.
            root_dir (str): Path to the root directory of the images.
        """

        self.img_root_dir = img_root_dir

        if transform:
            self.transform = transform 
        else:
            # Use timm's transformations instead of torch.compose, since they're targeted to ResNet50
            config = resolve_data_config({}, model='resnet50')
            self.transform  = create_transform(**config)        
        
        # Load captions and image paths from the JSON file
        with open(captions_file, 'r') as f:
            raw_coco_data = json.load(f)
            images = {img['id']: img['file_name'] for img in raw_coco_data['images']}
            data = []
            for ann in raw_coco_data['annotations']:
                if ann['image_id'] in images:
                    data.append({'image': images[ann['image_id']], 'caption': ann['caption']})
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_root_dir, sample['image'])
        image = Image.open(img_path).convert('RGB')
        
        caption = sample['caption']
        return image, caption

# ImageNet Validation Dataset for zero-shot predictions
class ImageNetDataset(Dataset):
    def __init__(self, img_root_dir, captions_file, transform=None):   
        """
        Args:
            img_root_dir (str): Path to the root directory of the images.
            captions_file (str): Path to the JSON file containing the ImageNet class index.
            transform (callable, optional): Transform to be applied to images.
        """
        self.img_root_dir = img_root_dir
        if transform:
            self.transform = transform 
        else:
            # Use timm's transformations instead of torch.compose, since they're targeted to ResNet50
            config = resolve_data_config({}, model='resnet50')
            self.transform  = create_transform(**config)     

        # Each directory in the ImageNet folder represents a class
        # These aren't human-readable, but the imagenet_class_index JSON file maps them to real world objects
        with open(captions_file, 'r') as f:
            self.class_index = json.load(f)
        
        # Convert to a format: {class_id: (imagenet_id, description)}
        idx_to_class = {int(k): v[1] for k, v in self.class_index.items()}

        dataset = datasets.ImageFolder(root=img_root_dir)
        self.data = []

        # Store image paths and their corresponding descriptions
        for path, label in dataset.samples:
            # Get the human-readable class name for the label
            class_description = idx_to_class[label]
            self.data.append({'image': path, 'description': class_description})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_root_dir, sample['image'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        description = sample['description']
        return image, description

# Perform image transformations and tokenization of raw captions
def collate_fn(batch, image_transform, tokenizer):
    images, captions = zip(*batch)
    images = torch.stack([image_transform(img) for img in images])
    cap_tokens = tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt")
    return images, cap_tokens

# Collate function for training set
def collate_fn_with_index(batch, image_transform, tokenizer):
    images, captions, indices = zip(*batch)
    images = torch.stack([image_transform(img) for img in images])
    cap_tokens = tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt")
    return images, cap_tokens, torch.tensor(indices, dtype=torch.long)

# Create DataLoader objects for the three datasets
def generate_loaders(parameters):
    CC3M_CAPTION_FILE = '../clip_train/cc3m_train_subset.json'
    CC3M_IMG_ROOT = '../datasets/cc3m_subset_100k'
    train = CC3MDataset(CC3M_CAPTION_FILE, CC3M_IMG_ROOT)

    COCO_CAPTION_FILE = '../datasets/mscoco_val/captions_val2014.json'
    COCO_IMG_ROOT = '../datasets/mscoco_val/mscoco_val2014_subset_5k'
    coco_valid = MSCOCODataset(COCO_CAPTION_FILE, COCO_IMG_ROOT)

    # NOTE: Run the shell script within imagenet first to get the val100 folder
    IMAGENET_IMG_ROOT = '../datasets/imagenet/val'
    IMAGENET_CAPTION_FILE = '../datasets/imagenet/imagenet_class_index.json'
    imagenet_valid = ImageNetDataset(IMAGENET_IMG_ROOT, IMAGENET_CAPTION_FILE)

    # NOTE: I'm assuming we want this consistent across datasets
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_loader = DataLoader(train, batch_size=parameters.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_with_index(batch, train.transform, tokenizer))
    coco_loader = DataLoader(coco_valid, batch_size=parameters.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=lambda batch: collate_fn(batch, coco_valid.transform, tokenizer))
    # No captions, use torch's default collate_fn
    imagenet_loader = DataLoader(imagenet_valid, batch_size=parameters.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, coco_loader, imagenet_loader



