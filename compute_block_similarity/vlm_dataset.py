import os
import pandas as pd

import torch 
from torch.utils.data import Dataset, DataLoader

class ShareGPT4VCOCODataset(Dataset):
    def __init__(self, num_samples=None):
        self.data_pairs = pd.read_csv('/home/haoli/VLM-prune/data/data_pairs.csv')
        self.img_dir = "/home/haoli/VLM-prune/data/train2017/train2017/"
        self.num_samples = num_samples
        
    def __len__(self):
        if self.num_samples is not None: 
            return self.num_samples
        return len(self.data_pairs)

    def __getitem__(self, idx):
        entry_id, caption = self.data_pairs.iloc[idx]
        img_file_path = os.path.join(self.img_dir, f"{entry_id}.jpg")
        return img_file_path, caption

if __name__ == "__main__":
    dataset = ShareGPT4VCOCODataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for img_file_path, caption in loader:
        print(img_file_path, caption)
        break
