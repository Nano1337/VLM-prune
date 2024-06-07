import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch 
from torch.utils.data import DataLoader, Dataset
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor

parquet_file_path = "tenk_filtered.parquet"
image_dir = "data/sampled_datacomp/sampled_datacomp_images"

from compute_block_similarity.mm_utils import expand2square

class TenkDataset(Dataset):
    def __init__(self):
        self.df = pd.read_parquet(parquet_file_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.iloc[idx]['uid']
        key = self.df.iloc[idx]['key']
        image_path = os.path.join(self.image_dir, key + ".jpg")
        image = Image.open(image_path).convert("RGB")
        # Ensure all images are square with a uniform size, e.g., 224x224
        image = expand2square(image, (255, 255, 255))  # Assuming white background
        image = image.resize((224, 224))  # Resize to a fixed dimension
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to tensor and change to (C, H, W) format
        caption = self.df.iloc[idx]['caption']
        return image_tensor, caption, uid
    
loader = DataLoader(TenkDataset(), batch_size=32, shuffle=False)
model, processor = _get_clip_model_and_processor(model_name_or_path="openai/clip-vit-large-patch14")

results = []

for batch in tqdm(loader, desc="Processing batches", unit="batch"):
    images, captions, uids = batch
    captions = list(captions)
    scores, _ = _clip_score_update(images, captions, model, processor)
    scores = scores / 100
    batch_results = list(zip(uids, scores.tolist()))
    results.extend(batch_results)
    del scores, batch_results

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['uid', 'CLIPscore'])

# Save the DataFrame to a Parquet file
df.to_parquet('clip_scores.parquet', index=False)

