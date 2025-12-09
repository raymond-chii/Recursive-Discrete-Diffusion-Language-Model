import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
MODEL_CONFIG = {
    'd_model': 516,
    'n_heads': 6,
    'n_layers': 6,
    'max_seq_len': 128,
    'dropout': 0.1
}

# Diffusion hyperparameters
DIFFUSION_CONFIG = {
    'num_timesteps': 50  # Reduced from 100 to minimize error accumulation during sampling
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 64,
    'num_epochs': 10,  # Reduced to 10 for faster training (can resume if needed)
    'learning_rate': 3e-4,  # Peak learning rate (will use warmup + decay)
    'warmup_steps': 2000,  # LR warmup for training stability
    'recursion_depth': 6,  # Reduced from 6 for ~2x speedup (still effective for diffusion)
    'gradient_clip': 5.0  # Increased from 1.0 - less restrictive, allows larger updates
}

# Dataset hyperparameters
# Model has 71.7M params → need 1.43B tokens (20 tokens/param)
# At 128 tokens/seq → need ~11.2M sequences
DATASET_CONFIG = {
    'dataset_name': 'HuggingFaceFW/fineweb-edu',  # High-quality educational web data
    'dataset_config': 'sample-10BT',  # 10B token sample (subset of full dataset)
    'max_length': 128,
    'num_samples': 12000000  # ~1.5B tokens to meet 20 tokens/param rule
}

# Sampling hyperparameters
SAMPLING_CONFIG = {
    'max_length': 128,  # Match training length
    'recursion_depth': 6,  # Match training depth for consistency
    'temperature': 1.0  # Slightly increased for more diversity
}
