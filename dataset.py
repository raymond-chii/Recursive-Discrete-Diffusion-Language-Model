import torch
import os
import shutil
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk, Dataset as HFDataset

# Hardcoded local path
LOCAL_DATA_PATH = "./fineweb_edu_sliced"

class TextDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=128, num_samples=5000000, 
                 dataset_name='HuggingFaceFW/fineweb-edu', dataset_config='sample-10BT', cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # --- SMART SLICED DOWNLOAD ---
        if not os.path.exists(LOCAL_DATA_PATH):
            print(f"\n[INFO] Local cache not found.")
            print(f"[DOWNLOAD] Downloading ONLY the first {num_samples} samples...")
            print("[DOWNLOAD] This will be small (~3-5GB) and fit on your disk.")
            
            try:
                # 1. Load in Streaming Mode
                ds_stream = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
                
                # 2. Take only what we need
                ds_slice = ds_stream.take(num_samples)
                
                # 3. CONVERT TO STATIC DATASET (The Fix)
                print("[DOWNLOAD] Materializing data... (This may take a few minutes)")
                
                def gen():
                    yield from ds_slice

                ds_static = HFDataset.from_generator(gen, features=ds_stream.features)
                
                # 4. Save to disk
                ds_static.save_to_disk(LOCAL_DATA_PATH)
                print(f"[DOWNLOAD] Success! Saved to {LOCAL_DATA_PATH}.\n")
                
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                # Clean up if failed
                if os.path.exists(LOCAL_DATA_PATH):
                    shutil.rmtree(LOCAL_DATA_PATH)
                raise e

        print(f"[FAST MODE] Loading data from local disk: {LOCAL_DATA_PATH}...")
        self.dataset = load_from_disk(LOCAL_DATA_PATH)
        
        # Convert to iterable for training consistency
        self.dataset = self.dataset.to_iterable_dataset()
        
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iterator = iter(self.dataset)
        else:
            iterator = iter(self.dataset)
            
        for example in iterator:
            text = example.get('text', '')
            if len(text) < 10: continue
            
            tokens = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_length
            )
            if len(tokens) > 2: 
                yield torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    pad_token_id = 50256
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded