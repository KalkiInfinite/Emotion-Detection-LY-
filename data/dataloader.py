"""
Data loaders and dataset classes for facial emotion recognition.
Supports multiple dataset formats including FER2013, AffectNet, and RAF-DB.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter


class EmotionDataset(Dataset):
    """
    Generic dataset class for facial emotion recognition.
    Supports various dataset formats.
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform=None,
                 emotion_labels: Optional[List[str]] = None):
        """
        Initialize emotion dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of emotion labels (integers)
            transform: Image transformations to apply
            emotion_labels: List of emotion label names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.emotion_labels = emotion_labels or ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        assert len(image_paths) == len(labels), "Number of images and labels must match"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        counter = Counter(self.labels)
        return {self.emotion_labels[i]: counter.get(i, 0) for i in range(len(self.emotion_labels))}


class FER2013Dataset(EmotionDataset):
    """Dataset class specifically for FER2013 dataset."""
    
    @classmethod
    def from_csv(cls, csv_path: str, image_dir: str, transform=None, split: str = None):
        """
        Create dataset from FER2013 CSV file.
        
        Args:
            csv_path: Path to FER2013 CSV file
            image_dir: Directory containing extracted images
            transform: Image transformations
            split: Data split ('train', 'val', 'test') - if None, uses all data
            
        Returns:
            FER2013Dataset instance
        """
        df = pd.read_csv(csv_path)
        
        if split:
            df = df[df['Usage'] == split]
        
        image_paths = []
        labels = []
        
        for idx, row in df.iterrows():
            image_filename = f"{idx}.jpg"  # Assuming images are saved as index.jpg
            image_path = os.path.join(image_dir, image_filename)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(int(row['emotion']))
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        return cls(image_paths, labels, transform, emotion_labels)


class DirectoryDataset(EmotionDataset):
    """
    Dataset class for datasets organized in directories by emotion.
    
    Expected structure:
    dataset_root/
    ├── angry/
    ├── happy/
    ├── sad/
    └── ...
    """
    
    @classmethod
    def from_directory(cls, root_dir: str, transform=None, valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
        """
        Create dataset from directory structure.
        
        Args:
            root_dir: Root directory containing emotion subdirectories
            transform: Image transformations
            valid_extensions: Valid image file extensions
            
        Returns:
            DirectoryDataset instance
        """
        image_paths = []
        labels = []
        emotion_labels = []
        
        # Get emotion directories
        emotion_dirs = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        emotion_dirs.sort()  # Ensure consistent ordering
        emotion_labels = emotion_dirs
        
        # Create label mapping
        label_map = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        
        # Collect images and labels
        for emotion_dir in emotion_dirs:
            emotion_path = os.path.join(root_dir, emotion_dir)
            emotion_label = label_map[emotion_dir]
            
            for filename in os.listdir(emotion_path):
                if filename.lower().endswith(valid_extensions):
                    image_path = os.path.join(emotion_path, filename)
                    image_paths.append(image_path)
                    labels.append(emotion_label)
        
        return cls(image_paths, labels, transform, emotion_labels)


def create_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted sampler for balanced training.
    
    Args:
        labels: List of labels in the dataset
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    
    # Calculate weights - inverse of class frequency
    weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Optional[Dataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       use_balanced_sampling: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_balanced_sampling: Whether to use balanced sampling for training
        
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader
    if use_balanced_sampling:
        sampler = create_balanced_sampler(train_dataset.labels)
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Validation loader
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def split_dataset(dataset: EmotionDataset, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Tuple[EmotionDataset, EmotionDataset, EmotionDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        dataset.image_paths, dataset.labels,
        test_size=(val_ratio + test_ratio),
        stratify=dataset.labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Create new datasets
    train_dataset = EmotionDataset(train_paths, train_labels, dataset.transform, dataset.emotion_labels)
    val_dataset = EmotionDataset(val_paths, val_labels, dataset.transform, dataset.emotion_labels)
    test_dataset = EmotionDataset(test_paths, test_labels, dataset.transform, dataset.emotion_labels)
    
    return train_dataset, val_dataset, test_dataset


def save_dataset_info(dataset: EmotionDataset, save_path: str):
    """Save dataset information to JSON file."""
    info = {
        'num_samples': len(dataset),
        'num_classes': len(dataset.emotion_labels),
        'emotion_labels': dataset.emotion_labels,
        'class_distribution': dataset.get_class_distribution()
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


def load_dataset_info(info_path: str) -> Dict[str, Any]:
    """Load dataset information from JSON file."""
    with open(info_path, 'r') as f:
        return json.load(f)
