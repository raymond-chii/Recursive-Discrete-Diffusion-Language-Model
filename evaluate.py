import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils import sample
from tqdm import tqdm

def measure_diversity(model, diffusion, tokenizer, num_samples=30, prompt="Once upon a time",
                      max_length=64, recursion_depth=6, temperature=0.9, device='cuda'):
    print(f"Generating {num_samples} samples to measure diversity...\n")

    samples = []
    for i in range(num_samples):
        text = sample(
            model=model,
            diffusion=diffusion,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            recursion_depth=recursion_depth,
            temperature=temperature,
            device=device
        )
        samples.append(text)

    all_tokens = []
    for text in samples:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    print(f"\nDiversity Metrics:")
    print(f"  Unique tokens: {unique_tokens} / {total_tokens}")
    print(f"  Type-Token Ratio: {ttr:.3f}")

    return samples


def compute_perplexity(model, diffusion, tokenizer, num_test_samples=200, device='cuda', dataset_name=None, dataset_config=None):
    print(f"Loading test data ({dataset_name})...")
    
    # Robust Dataset Loading
    try:
        if dataset_name == 'wikitext':
            test_dataset = load_dataset(dataset_name, dataset_config, split='test', streaming=True)
        elif 'fineweb' in str(dataset_name).lower():
            # Skip training data to simulate test set if using same dataset
            test_dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
            test_dataset = test_dataset.skip(200000) 
        else:
            test_dataset = load_dataset(dataset_name, split='train', streaming=True).skip(1000)
    except Exception as e:
        print(f"Warning: Could not load {dataset_name} ({e}). using OpenWebText fallback.")
        test_dataset = load_dataset('openwebtext', split='train', streaming=True).skip(1000)

    test_samples = []
    iterator = iter(test_dataset)
    for _ in range(num_test_samples):
        try:
            example = next(iterator)
            test_samples.append(example['text'])
        except StopIteration:
            break

    print(f"Loaded {len(test_samples)} test samples")

    model.eval()
    total_loss = 0
    total_tokens = 0
    fixed_timestep = 25 # Check at t=25 (50% noise)

    with torch.no_grad():
        for text in tqdm(test_samples, desc=f"Eval {dataset_name}"):
            if not text.strip(): continue
            
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            if tokens.size(1) < 5: continue

            t = torch.tensor([fixed_timestep], device=device)
            x_t, mask = diffusion.q_sample(tokens, t)

            if not mask.any(): continue

            # FIX: Model returns LIST of logits. Get the last one.
            all_logits = model(x_t, t, recursion_depth=4)
            logits = all_logits[-1] 

            loss = F.cross_entropy(
                logits[mask].reshape(-1, logits.size(-1)),
                tokens[mask].reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += mask.sum().item()

    perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    print(f"\n=== Diffusion Perplexity (t={fixed_timestep}) ===")
    print(f"  Dataset: {dataset_name}")
    print(f"  Score: {perplexity:.2f}")

    return perplexity


def evaluate_generation_quality_phi2(samples, device='cuda', plot_histogram=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading Evaluation Model (GPT-2 Large)...")
    # Using GPT-2 Large is safer for memory than Phi-2
    eval_model_name = 'gpt2-large'
    
    try:
        eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    except Exception as e:
        print(f"Failed to load evaluator: {e}")
        return 0.0, []

    eval_model.eval()
    perplexities = []
    
    with torch.no_grad():
        for i, text in enumerate(samples):
            if not text.strip(): continue
            tokens = eval_tokenizer.encode(text, return_tensors='pt').to(device)
            if tokens.size(1) < 2: continue

            outputs = eval_model(tokens, labels=tokens)
            ppl = torch.exp(outputs.loss).item()
            
            if ppl < 500: # Filter garbage outliers
                perplexities.append(ppl)

    avg_ppl = np.mean(perplexities) if perplexities else 0.0
    print(f"  Average Quality Score (PPL): {avg_ppl:.2f}")

    if plot_histogram and perplexities:
        plt.figure(figsize=(8, 5))
        plt.hist(perplexities, bins=10, color='skyblue', edgecolor='black')
        plt.title('Generation Quality Distribution')
        plt.xlabel('GPT-2 Perplexity (Lower is Better)')
        plt.savefig('generation_quality_hist.png')

    return avg_ppl, perplexities