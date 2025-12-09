import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

class TextDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=128, num_samples=10000000, 
                 dataset_name='openwebtext', dataset_config=None, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        print(f"Loading {dataset_name} in streaming mode...")
        
        if dataset_config:
            self.dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split='train', streaming=True)
            
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        
        self.dataset = self.dataset.take(num_samples)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            iterator = iter(self.dataset)
        else:

            iterator = iter(self.dataset)
            
        for example in iterator:
            text = example.get('text', '')
            
            if len(text) < 10: 
                continue
                
            # Tokenize on-the-fly
            tokens = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_length
            )
            
            # Yield only valid sequences
            if len(tokens) > 2: 
                yield torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    # Standard padding logic
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded