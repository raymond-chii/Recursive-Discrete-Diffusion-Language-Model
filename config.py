import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
MODEL_CONFIG = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 3,     
    'max_seq_len': 128,
    'dropout': 0.1
}

# Diffusion hyperparameters
DIFFUSION_CONFIG = {
    'num_timesteps': 50
}

# Training hyperparameters (Optimized for A100)
TRAINING_CONFIG = {
    'batch_size': 128,
    'accum_steps': 2,
    'num_epochs': 1,
    'learning_rate': 1e-4,  # Stable for recursion
    'warmup_steps': 2000, 
    'recursion_depth': 6,
    'gradient_clip': 1.0,
    'use_tf32': True,       # Speed boost for Ampere GPUs
    'mixed_precision': 'bf16' # Critical for stability
}

# Dataset hyperparameters
DATASET_CONFIG = {
    'dataset_name': 'HuggingFaceFW/fineweb-edu',
    'dataset_config': 'sample-10BT',
    'max_length': 128,
    'num_samples': 5000000  # 5 Million samples
}

# Sampling hyperparameters
SAMPLING_CONFIG = {
    'max_length': 128,
    'recursion_depth': 6,
    'temperature': 1.0
}