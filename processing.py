import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
from torchvision import transforms

# path
DATA_DIR = "/path/to/train_images"  # adjust
CSV_PATH = "train.csv"

# load csv
df = pd.read_csv(CSV_PATH)

# for ConvNeXt
preprocess = transforms.Compose([
    # convet to array
    transforms.ToPILImage(),
    # what sizing?
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# load and preprocess DICOM image
def load_dicom_image(image_id, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{image_id}.dcm")
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    
    # normalize 0-255
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # apply transforms
    img_tensor = preprocess(img)
    return img_tensor

# ex: load first 5 images and their metadata
for idx, row in df.head(5).iterrows():
    img_tensor = load_dicom_image(row['image_id'])
    label = torch.tensor(row['cancer'], dtype=torch.float32)
    
    # optional: simple metadata vector
    metadata = torch.tensor([row['age'], row['density']], dtype=torch.float32)
    
    print(f"Loaded {row['image_id']}, tensor shape: {img_tensor.shape}, label: {label}, metadata: {metadata}")
