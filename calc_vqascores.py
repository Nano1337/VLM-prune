import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


import torch 
from torch.utils.data import DataLoader, Dataset
from compute_block_similarity.clip_t5_model import CLIPT5Model

parquet_file_path = "data/sampled_datacomp/tenk_dataset.parquet"
image_dir = "data/sampled_datacomp/sampled_datacomp_images"

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
        caption = self.df.iloc[idx]['caption']
        return image_path, caption, uid


batch_size = 24

dataset = TenkDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = CLIPT5Model(calc_score=True)

results = []

for batch in tqdm(loader, desc="Processing batches", unit="batch"):
    image_paths, captions, uids = batch
    try:
        scores = model.forward(image_paths, captions)
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Image paths that caused the error:")
        for path in image_paths:
            print(path)
            try:
                file_size = os.path.getsize(path)
                print(f"File size: {file_size} bytes")
            except OSError as e:
                print(f"Could not get file size for {path}: {e}")
        scores = torch.ones(len(image_paths)) * -1 # filter out these invalid samples later
    batch_results = list(zip(uids, scores.tolist()))
    results.extend(batch_results)
    del scores, batch_results

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['uid', 'VQAscore'])

# Save the DataFrame to a Parquet file
df.to_parquet('vqa_scores.parquet', index=False)

"""
Known Issues: 
- It seems that the model randomly goes OOM sometimes. I'm not quite sure what's causing it, but precomputing CLIP embeddings should help
    - it seems more likely that there are extremely narrow images that when expand2square() is called, the resulting image is massive
"""

