"""
Configuration settings for ViT emotion recognition model.
"""

# Model configuration
MODEL_CONFIG = {
    'model_name': 'google/vit-base-patch16-224',  # Pretrained ViT model
    'num_classes': 7,                             # Number of emotion classes
    'dropout_rate': 0.1,                          # Dropout rate for classifier
    'freeze_backbone': False,                     # Whether to freeze ViT backbone
    'image_size': 224,                            # Input image size
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'scheduler': 'cosine',                        # Learning rate scheduler
    'warmup_epochs': 5,                           # Warmup epochs for scheduler
    'gradient_clip_norm': 1.0,                    # Gradient clipping
    'use_mixed_precision': True,                  # Use automatic mixed precision
    'accumulation_steps': 1,                      # Gradient accumulation steps
}

# Data configuration
DATA_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'use_balanced_sampling': True,                # Use balanced sampling for training
    'num_workers': 4,                             # DataLoader workers
    'pin_memory': True,                           # Pin memory for faster GPU transfer
    'augmentation': {
        'use_augmentation': True,
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 10,
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'gaussian_noise_std': 0.005,
    }
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    'fer2013': {
        'num_classes': 7,
        'emotion_labels': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'dataset_type': 'csv'
    },
    'affectnet': {
        'num_classes': 8,
        'emotion_labels': ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
        'dataset_type': 'directory'
    },
    'rafdb': {
        'num_classes': 7,
        'emotion_labels': ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral'],
        'dataset_type': 'directory'
    }
}

# Paths configuration
PATHS_CONFIG = {
    'data_dir': './data',
    'output_dir': './outputs',
    'checkpoint_dir': './outputs/checkpoints',
    'log_dir': './outputs/logs',
    'evaluation_dir': './outputs/evaluation',
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'save_frequency': 5,                          # Save checkpoint every N epochs
    'validation_frequency': 1,                    # Validate every N epochs
    'tensorboard_logging': True,
    'console_logging': True,
    'log_metrics': ['accuracy', 'f1_score', 'precision', 'recall']
}

# Hardware configuration
HARDWARE_CONFIG = {
    'device': 'auto',                             # 'auto', 'mps', 'cuda', 'cpu'
    'prefer_mps': True,                           # Prefer MPS on Apple Silicon
    'multi_gpu': False,                           # Use multiple GPUs if available
    'mixed_precision': True,                      # Use mixed precision training (Note: MPS has limited support)
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
    'save_predictions': True,
    'save_probabilities': True,
    'generate_plots': True,
    'plot_formats': ['png', 'pdf'],
}

# Inference configuration
INFERENCE_CONFIG = {
    'batch_size': 32,
    'use_tta': False,                             # Test-time augmentation
    'confidence_threshold': 0.5,                  # Minimum confidence for prediction
    'webcam_settings': {
        'frame_width': 640,
        'frame_height': 480,
        'fps': 30,
        'face_detection_scale_factor': 1.1,
        'face_detection_min_neighbors': 5,
        'face_detection_min_size': (30, 30),
    }
}
