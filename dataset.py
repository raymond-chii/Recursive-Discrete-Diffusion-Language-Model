import torch
import os
import pickle
from torch.utils.data import Dataset
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, num_samples=10000000, dataset_name='openwebtext',
                 dataset_config=None, cache_dir='./data_cache'):
        # Create cache filename
        os.makedirs(cache_dir, exist_ok=True)
        config_str = f"_{dataset_config}" if dataset_config else ""
        cache_file = os.path.join(cache_dir, f'{dataset_name.replace("/", "_")}{config_str}_{num_samples}_{max_length}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.encodings = pickle.load(f)
            print(f"Loaded {len(self.encodings)} cached sequences")
        else:
            # Load and process dataset
            print(f"Loading {dataset_name} dataset from HuggingFace...")
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
            else:
                dataset = load_dataset(dataset_name, split='train', streaming=True)

            self.encodings = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                # WikiText uses 'text', OpenWebText uses 'text'
                text = example.get('text', '')

                # Skip empty or very short texts
                if len(text.strip()) < 10:
                    continue

                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=max_length
                )
                if len(tokens) > 10:
                    self.encodings.append(tokens)

                if i % 1000 == 0:
                    print(f"Processed {i}/{num_samples} samples, collected {len(self.encodings)} sequences")

            print(f"Total sequences: {len(self.encodings)}")

            # Save to cache (with error handling for disk space issues)
            try:
                print(f"Saving dataset to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.encodings, f)
                print("Cache saved!")
            except OSError as e:
                print(f"⚠️  WARNING: Could not save cache due to disk space: {e}")
                print("Continuing without cache - dataset will need to be reloaded on next run")
            except Exception as e:
                print(f"⚠️  WARNING: Unexpected error saving cache: {e}")
                print("Continuing without cache")

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
