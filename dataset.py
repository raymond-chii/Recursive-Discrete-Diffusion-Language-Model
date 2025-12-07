import torch
import os
import pickle
from torch.utils.data import Dataset
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, num_samples=10000000, dataset_name='openwebtext', cache_dir='./data_cache'):
        # Create cache filename
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{dataset_name.replace("/", "_")}_{num_samples}_{max_length}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.encodings = pickle.load(f)
            print(f"Loaded {len(self.encodings)} cached sequences")
        else:
            # Load and process dataset
            print(f"Loading {dataset_name} dataset from HuggingFace (this will take 30-60 min)...")
            dataset = load_dataset(dataset_name, split='train', streaming=True)

            self.encodings = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                tokens = tokenizer.encode(
                    example['text'],
                    truncation=True,
                    max_length=max_length
                )
                if len(tokens) > 10:
                    self.encodings.append(tokens)

                if i % 1000 == 0:
                    print(f"Loaded {i}/{num_samples} samples")

            print(f"Total sequences: {len(self.encodings)}")

            # Save to cache
            print(f"Saving dataset to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.encodings, f)
            print("Cache saved!")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded
