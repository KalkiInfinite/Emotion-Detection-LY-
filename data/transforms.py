"""
Data transformations for facial emotion recognition.
Includes augmentations and preprocessing for ViT models.
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from PIL import Image, ImageEnhance
import numpy as np
from typing import Tuple, Optional


class FaceAugmentation:
    """Custom augmentation specifically designed for facial emotion recognition."""
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image):
        # Random brightness adjustment
        brightness_factor = random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Random contrast adjustment
        contrast_factor = random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        return image


class GaussianNoise:
    """Add Gaussian noise to images."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


def get_train_transforms(image_size: int = 224, 
                        use_augmentation: bool = True) -> transforms.Compose:
    """
    Get training data transformations.
    
    Args:
        image_size: Target image size for ViT (default: 224)
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        Composed transforms for training data
    """
    transform_list = []
    
    # Resize with some padding to allow for rotation/crop
    if use_augmentation:
        transform_list.extend([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            FaceAugmentation(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    # Add noise as final augmentation
    if use_augmentation:
        transform_list.append(GaussianNoise(std=0.005))
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transformations.
    
    Args:
        image_size: Target image size for ViT (default: 224)
        
    Returns:
        Composed transforms for validation/test data
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get inference transformations (same as validation).
    
    Args:
        image_size: Target image size for ViT (default: 224)
        
    Returns:
        Composed transforms for inference
    """
    return get_val_transforms(image_size)


class TenCropTransform:
    """
    Apply TenCrop transformation for test-time augmentation.
    Useful for more robust inference.
    """
    
    def __init__(self, size: int = 224):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((int(size * 1.2), int(size * 1.2))),
            transforms.TenCrop(size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(crop) for crop in crops
            ]))
        ])
    
    def __call__(self, image):
        return self.transform(image)


# Emotion class mappings
EMOTION_LABELS = {
    'fer2013': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    'affectnet': ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
    'rafdb': ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
}

def get_emotion_labels(dataset_name: str = 'fer2013') -> list:
    """
    Get emotion labels for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('fer2013', 'affectnet', 'rafdb')
        
    Returns:
        List of emotion label names
    """
    return EMOTION_LABELS.get(dataset_name.lower(), EMOTION_LABELS['fer2013'])


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return tensor * std + mean


