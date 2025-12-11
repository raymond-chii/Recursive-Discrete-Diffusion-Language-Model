import torch
from transformers import GPT2Tokenizer

from model import TinyRecursiveTransformer, AbsorbingDiffusion
from utils import sample
from config import device


def demo_generation():

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    mask_token_id = len(tokenizer)
    vocab_size = len(tokenizer) + 1

    # Load model
    checkpoint = torch.load('trm_diffusion.pt', map_location=device)
    model = TinyRecursiveTransformer(
        vocab_size=vocab_size,
        d_model=516,
        n_heads=6,
        n_layers=6,
        max_seq_len=128,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load diffusion
    diffusion = AbsorbingDiffusion(
        num_timesteps=1000,
        mask_token_id=mask_token_id
    )

    # Test prompts (suitable for OpenWebText)
    prompts = [
        "The future of technology",
        "In recent years",
        "According to experts",
        "Many people believe"
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print('='*60)

        for temp in [0.7, 1.0]:
            text = sample(
                model=model,
                diffusion=diffusion,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=128,
                recursion_depth=10,
                temperature=temp,
                device=device
            )
            print(f"\nTemp {temp}: {text}")


if __name__ == "__main__":
    demo_generation()