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
    'num_timesteps': 50
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 512,
    'num_epochs': 1,
    'learning_rate': 6e-4,
    'warmup_steps': 1000, 
    'recursion_depth': 6,
    'gradient_clip': 5.0
}

# Dataset hyperparameters
DATASET_CONFIG = {
    'dataset_name': 'HuggingFaceFW/fineweb-edu',
    'dataset_config': 'sample-10BT',
    'max_length': 128,
    'num_samples': 12000000 
}

# Sampling hyperparameters
SAMPLING_CONFIG = {
    'max_length': 128,
    'recursion_depth': 6,
    'temperature': 1.0
}