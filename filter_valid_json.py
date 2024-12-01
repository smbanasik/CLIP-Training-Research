import json
import os

# The captions_val2014.json file from the Google Drive contains data for the entire MSCOCO14 dataset
# We only use a subset of these, so some images are missing
# We need to filter out all unused images as to not break the dataset constructor 

# Paths to files and directories
json_file = './Datasets/mscoco_val/captions_val2014.json'
image_dir = './Datasets/mscoco_val/mscoco_val2014_subset_5k'
output_file = './Datasets/mscoco_val/filtered_captions_val2014.json'

# Get available image filenames
available_images = set(os.listdir(image_dir))

# Load JSON data
with open(json_file, 'r') as f:
    coco_data = json.load(f)

# Filter images and annotations
filtered_images = []
filtered_annotations = []
for img in coco_data['images']:
    if img['file_name'] in available_images:
        filtered_images.append(img)

valid_image_ids = {img['id'] for img in filtered_images}
for ann in coco_data['annotations']:
    if ann['image_id'] in valid_image_ids:
        filtered_annotations.append(ann)

# Save filtered JSON
filtered_data = {
    'images': filtered_images,
    'annotations': filtered_annotations
}
with open(output_file, 'w') as f:
    json.dump(filtered_data, f)

print(f"Filtered captions saved to {output_file}")
