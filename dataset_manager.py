"""
Multi-Dataset Manager for Facial Emotion Recognition
This script helps you download, organize, and combine multiple emotion datasets.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter

# Dataset configurations
DATASETS_CONFIG = {
    'fer2013': {
        'name': 'FER2013',
        'size': '150 MB',
        'images': 35887,
        'classes': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'urls': [
            'https://www.kaggle.com/datasets/msambare/fer2013',
            'https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge'
        ],
        'format': 'directory',
        'recommended': True
    },
    'affectnet': {
        'name': 'AffectNet',
        'size': '22 GB',
        'images': 400000,
        'classes': ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
        'urls': [
            'http://mohammadmahoor.com/affectnet/',
            'https://www.kaggle.com/datasets/noamsegal/affectnet-training-data'
        ],
        'format': 'directory',
        'recommended': True
    },
    'rafdb': {
        'name': 'RAF-DB',
        'size': '2 GB',
        'images': 29672,
        'classes': ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral'],
        'urls': [
            'http://www.whdeng.cn/raf/model1.html',
            'https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset'
        ],
        'format': 'directory',
        'recommended': True
    },
    'ferplus': {
        'name': 'FERPlus',
        'size': '150 MB',
        'images': 35887,
        'classes': ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'],
        'urls': [
            'https://github.com/microsoft/FERPlus',
            'https://www.kaggle.com/datasets/gauravsharma99/fer2013plus'
        ],
        'format': 'directory',
        'recommended': True
    },
    'jaffe': {
        'name': 'JAFFE',
        'size': '5 MB',
        'images': 213,
        'classes': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'urls': [
            'https://zenodo.org/record/3451524',
            'https://www.kaggle.com/datasets/rizdelhi/jaffe-dataset'
        ],
        'format': 'directory',
        'recommended': False
    },
    'ckplus': {
        'name': 'CK+',
        'size': '50 MB',
        'images': 981,
        'classes': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'urls': [
            'http://www.cs.cmu.edu/afs/cs/project/face/www/facs.htm',
            'https://www.kaggle.com/datasets/shawon10/ckplus'
        ],
        'format': 'directory',
        'recommended': False
    },
    'kdef': {
        'name': 'KDEF',
        'size': '200 MB',
        'images': 4900,
        'classes': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'urls': [
            'https://www.emotionlab.se/kdef/'
        ],
        'format': 'directory',
        'recommended': False
    }
}

class MultiDatasetManager:
    """Manager for multiple emotion recognition datasets."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.combined_dir = self.base_dir / "combined"
        self.stats = {}
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.combined_dir.mkdir(exist_ok=True)
    
    def print_dataset_info(self):
        """Print information about all available datasets."""
        print("🎯 FACIAL EMOTION RECOGNITION DATASETS")
        print("=" * 80)
        
        print("\n🔥 RECOMMENDED DATASETS (Download these first):")
        recommended = [(k, v) for k, v in DATASETS_CONFIG.items() if v['recommended']]
        
        for dataset_id, info in recommended:
            print(f"\n📊 {info['name']} ({dataset_id.upper()})")
            print(f"   Size: {info['size']} | Images: {info['images']:,}")
            print(f"   Classes: {len(info['classes'])} emotions")
            print(f"   Download links:")
            for url in info['urls']:
                print(f"     • {url}")
        
        print(f"\n⭐ ADDITIONAL DATASETS (Optional):")
        additional = [(k, v) for k, v in DATASETS_CONFIG.items() if not v['recommended']]
        
        for dataset_id, info in additional:
            print(f"\n📊 {info['name']} ({dataset_id.upper()})")
            print(f"   Size: {info['size']} | Images: {info['images']:,}")
            print(f"   Download links:")
            for url in info['urls']:
                print(f"     • {url}")
        
        print(f"\n💡 DOWNLOAD INSTRUCTIONS:")
        print("1. Click on Kaggle links (most reliable)")
        print("2. Download and extract to: ./data/[dataset_name]/")
        print("3. Run this script again to combine datasets")
        print("4. Expected structure:")
        print("   data/")
        print("   ├── fer2013/train/angry/, happy/, ...")
        print("   ├── affectnet/train/neutral/, happiness/, ...")
        print("   ├── rafdb/train/anger/, joy/, ...")
        print("   └── combined/ (auto-generated)")
    
    def check_available_datasets(self) -> Dict[str, bool]:
        """Check which datasets are available locally."""
        available = {}
        
        print("\n🔍 CHECKING AVAILABLE DATASETS:")
        print("-" * 50)
        
        for dataset_id, info in DATASETS_CONFIG.items():
            dataset_path = self.base_dir / dataset_id
            
            if dataset_path.exists():
                # Count images
                image_count = sum(1 for p in dataset_path.rglob("*.jpg")) + \
                             sum(1 for p in dataset_path.rglob("*.png")) + \
                             sum(1 for p in dataset_path.rglob("*.jpeg"))
                
                if image_count > 0:
                    available[dataset_id] = True
                    print(f"✅ {info['name']:12} | {image_count:6,} images found")
                    self.stats[dataset_id] = image_count
                else:
                    available[dataset_id] = False
                    print(f"❌ {info['name']:12} | Directory exists but no images")
            else:
                available[dataset_id] = False
                print(f"❌ {info['name']:12} | Not found")
        
        total_images = sum(self.stats.values())
        print(f"\n📊 TOTAL AVAILABLE IMAGES: {total_images:,}")
        
        return available
    
    def normalize_emotion_labels(self, dataset_id: str, original_label: str) -> str:
        """Normalize emotion labels across datasets to standard 7 emotions."""
        
        # Standard 7 emotions we'll use
        standard_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Mapping from various dataset labels to standard labels
        label_mapping = {
            # Common variations
            'anger': 'angry',
            'happiness': 'happy',
            'joy': 'happy',
            'sadness': 'sad',
            
            # AffectNet specific
            'contempt': 'disgust',  # Map contempt to disgust as they're similar
            
            # RAF-DB might have different naming
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'angry': 'angry',
            'surprise': 'surprise'
        }
        
        # Normalize to lowercase first
        normalized = original_label.lower().strip()
        
        # Apply mapping if exists
        if normalized in label_mapping:
            return label_mapping[normalized]
        
        # If already in standard emotions, return as is
        if normalized in standard_emotions:
            return normalized
        
        # Default fallback
        print(f"⚠️ Unknown emotion '{original_label}' in {dataset_id}, mapping to 'neutral'")
        return 'neutral'
    
    def combine_datasets(self, selected_datasets: List[str] = None) -> bool:
        """Combine multiple datasets into a unified format."""
        
        if selected_datasets is None:
            available = self.check_available_datasets()
            selected_datasets = [k for k, v in available.items() if v]
        
        if not selected_datasets:
            print("❌ No datasets available to combine!")
            return False
        
        print(f"\n🔄 COMBINING DATASETS: {', '.join(selected_datasets)}")
        print("-" * 50)
        
        # Create combined directory structure
        standard_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        for split in ['train', 'val', 'test']:
            for emotion in standard_emotions:
                (self.combined_dir / split / emotion).mkdir(parents=True, exist_ok=True)
        
        total_copied = 0
        emotion_counts = Counter()
        
        for dataset_id in selected_datasets:
            print(f"\n📂 Processing {dataset_id.upper()}...")
            dataset_path = self.base_dir / dataset_id
            
            if not dataset_path.exists():
                print(f"   ⚠️ Skipping {dataset_id} - not found")
                continue
            
            # Find all images in the dataset
            image_extensions = ['.jpg', '.jpeg', '.png']
            dataset_images = []
            
            for ext in image_extensions:
                dataset_images.extend(dataset_path.rglob(f"*{ext}"))
                dataset_images.extend(dataset_path.rglob(f"*{ext.upper()}"))
            
            copied_count = 0
            
            for img_path in dataset_images:
                try:
                    # Try to determine emotion from path
                    emotion = None
                    path_parts = img_path.parts
                    
                    # Look for emotion in path
                    for part in path_parts:
                        part_lower = part.lower()
                        if any(emo in part_lower for emo in ['angry', 'anger', 'disgust', 'fear', 'happy', 'joy', 'happiness', 'neutral', 'sad', 'sadness', 'surprise']):
                            emotion = self.normalize_emotion_labels(dataset_id, part)
                            break
                    
                    if emotion is None:
                        continue  # Skip if can't determine emotion
                    
                    # Determine split (prefer train for most images)
                    if 'test' in str(img_path).lower():
                        split = 'test'
                    elif 'val' in str(img_path).lower():
                        split = 'val'
                    else:
                        split = 'train'  # Default to train
                    
                    # Create new filename with dataset prefix
                    new_filename = f"{dataset_id}_{img_path.stem}_{copied_count}{img_path.suffix}"
                    dest_path = self.combined_dir / split / emotion / new_filename
                    
                    # Copy image
                    shutil.copy2(img_path, dest_path)
                    copied_count += 1
                    emotion_counts[emotion] += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error copying {img_path}: {e}")
            
            print(f"   ✅ Copied {copied_count:,} images from {dataset_id}")
            total_copied += copied_count
        
        print(f"\n🎉 COMBINATION COMPLETE!")
        print(f"📊 Total images combined: {total_copied:,}")
        print(f"\n🎯 Emotion distribution:")
        for emotion, count in emotion_counts.most_common():
            print(f"   {emotion:>8}: {count:6,} images")
        
        # Save combination stats
        stats = {
            'total_images': total_copied,
            'datasets_used': selected_datasets,
            'emotion_distribution': dict(emotion_counts),
            'created_at': str(pd.Timestamp.now())
        }
        
        with open(self.combined_dir / 'dataset_info.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📁 Combined dataset location: {self.combined_dir}")
        print(f"📈 Dataset info saved to: {self.combined_dir}/dataset_info.json")
        
        return True
    
    def get_training_command(self) -> str:
        """Get the training command for the combined dataset."""
        return f"""
🚀 TRAINING COMMAND FOR COMBINED DATASET:

cd /Users/piyushtyagi/FacialEmotion

# Train with combined dataset (recommended settings)
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train.py \\
    --data_dir data/combined \\
    --dataset_type directory \\
    --num_epochs 50 \\
    --batch_size 32 \\
    --learning_rate 3e-5 \\
    --save_dir outputs/combined_model

# For faster training on large dataset
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train.py \\
    --data_dir data/combined \\
    --dataset_type directory \\
    --num_epochs 30 \\
    --batch_size 64 \\
    --learning_rate 1e-4 \\
    --save_dir outputs/combined_model
"""

def main():
    """Main function."""
    print("🎯 MULTI-DATASET EMOTION RECOGNITION MANAGER")
    print("=" * 80)
    
    manager = MultiDatasetManager()
    
    print("\n📋 Choose an option:")
    print("1. Show all available datasets and download links")
    print("2. Check which datasets are already downloaded")
    print("3. Combine available datasets")
    print("4. Show training command for combined dataset")
    print("5. All of the above")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice in ['1', '5']:
            manager.print_dataset_info()
        
        if choice in ['2', '5']:
            manager.check_available_datasets()
        
        if choice in ['3', '5']:
            print("\n" + "=" * 50)
            available = manager.check_available_datasets()
            available_datasets = [k for k, v in available.items() if v]
            
            if available_datasets:
                manager.combine_datasets(available_datasets)
            else:
                print("❌ No datasets available to combine. Download some datasets first!")
        
        if choice in ['4', '5']:
            print(manager.get_training_command())
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()
