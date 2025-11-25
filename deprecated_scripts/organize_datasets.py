"""
Dataset Organization and Conversion Script
Converts all your downloaded datasets into a unified format for training.
"""

import os
import shutil
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List
import json
from collections import Counter
import cv2
from PIL import Image

class DatasetOrganizer:
    """Organizes and converts multiple dataset formats into unified structure."""
    
    def __init__(self, base_dir: str = "/Users/piyushtyagi/FacialEmotion"):
        self.base_dir = Path(base_dir)
        self.unified_dir = self.base_dir / "data" / "unified_dataset"
        
        # Standard emotion mapping
        self.emotion_mapping = {
            # Lowercase standardization
            'anger': 'angry',
            'angry': 'angry',
            'contempt': 'contempt',  # Keep separate for now
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'happiness': 'happy',
            'joy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'sadness': 'sad',
            'surprise': 'surprise',
            
            # Numeric mappings (common in datasets)
            '0': 'angry',
            '1': 'disgust', 
            '2': 'fear',
            '3': 'happy',
            '4': 'sad',
            '5': 'surprise',
            '6': 'neutral',
            '7': 'contempt'
        }
    
    def analyze_datasets(self):
        """Analyze all available datasets."""
        print("🔍 ANALYZING YOUR DATASETS")
        print("=" * 60)
        
        datasets = {
            'archive': self.base_dir / 'archive',
            'archive_2': self.base_dir / 'archive (2)',
            'archive_3': self.base_dir / 'archive (3)', 
            'yolo_format': self.base_dir / 'YOLO_format'
        }
        
        total_images = 0
        
        for name, path in datasets.items():
            if path.exists():
                # Count images
                image_count = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.png")))
                print(f"\n📁 {name.upper().replace('_', ' ')}")
                print(f"   Location: {path}")
                print(f"   Images: {image_count:,}")
                
                # Analyze structure
                if name == 'archive':
                    print("   Format: FER2013 directory structure ✅")
                    print("   Emotions: 7 (angry, disgust, fear, happy, neutral, sad, surprise)")
                    
                elif name == 'archive_2':
                    print("   Format: CSV + Images 🔄 (needs conversion)")
                    if (path / 'train_labels.csv').exists():
                        df = pd.read_csv(path / 'train_labels.csv')
                        print(f"   CSV entries: {len(df):,}")
                    
                elif name == 'archive_3':
                    print("   Format: Directory structure ✅")
                    emotions = [d.name for d in (path / 'train').iterdir() if d.is_dir()]
                    print(f"   Emotions: {len(emotions)} ({', '.join(emotions)})")
                    
                elif name == 'yolo_format':
                    print("   Format: YOLO format 🔄 (needs conversion)")
                    if (path / 'data.yaml').exists():
                        with open(path / 'data.yaml', 'r') as f:
                            yolo_config = yaml.safe_load(f)
                        print(f"   Classes: {yolo_config.get('nc', 'unknown')}")
                        print(f"   Names: {yolo_config.get('names', [])}")
                
                total_images += image_count
            else:
                print(f"\n❌ {name.upper()}: Not found")
        
        print(f"\n🎯 TOTAL IMAGES: {total_images:,}")
        print("📊 This is an EXCELLENT dataset size for training!")
        
        return total_images
    
    def convert_yolo_format(self):
        """Convert YOLO format to directory structure."""
        print("\n🔄 CONVERTING YOLO FORMAT...")
        
        yolo_dir = self.base_dir / 'YOLO_format' 
        if not yolo_dir.exists():
            print("❌ YOLO format directory not found")
            return 0
        
        # Read YOLO config
        config_file = yolo_dir / 'data.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                yolo_config = yaml.safe_load(f)
            class_names = yolo_config.get('names', [])
        else:
            # Default AffectNet classes
            class_names = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        
        print(f"   Classes: {class_names}")
        
        converted_count = 0
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            split_name = 'val' if split == 'valid' else split
            
            images_dir = yolo_dir / split / 'images'
            labels_dir = yolo_dir / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            print(f"   Processing {split}...")
            
            # Process each image
            for img_file in images_dir.glob("*.png"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    try:  
                        # Read YOLO label (format: class_id x_center y_center width height)
                        with open(label_file, 'r') as f:
                            label_content = f.read().strip()
                        
                        if label_content:
                            class_id = int(label_content.split()[0])
                            if class_id < len(class_names):
                                emotion_name = class_names[class_id].lower()
                                emotion_name = self.emotion_mapping.get(emotion_name, emotion_name)
                                
                                # Create destination directory
                                dest_dir = self.unified_dir / split_name / emotion_name
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Copy image
                                dest_file = dest_dir / f"yolo_{img_file.name}"
                                shutil.copy2(img_file, dest_file)
                                converted_count += 1
                    
                    except Exception as e:
                        print(f"   ⚠️ Error processing {img_file.name}: {e}")
        
        print(f"   ✅ Converted {converted_count:,} images from YOLO format")
        return converted_count
    
    def convert_csv_format(self):
        """Convert CSV format (Archive 2) to directory structure."""  
        print("\n🔄 CONVERTING CSV FORMAT...")
        
        csv_dir = self.base_dir / 'archive (2)'
        if not csv_dir.exists():
            print("❌ CSV format directory not found")
            return 0
        
        converted_count = 0
        
        # Process train and test CSV files
        for csv_name in ['train_labels.csv', 'test_labels.csv']:
            csv_file = csv_dir / csv_name
            if not csv_file.exists():
                continue
                
            split_name = 'train' if 'train' in csv_name else 'test'
            print(f"   Processing {csv_name}...")
            
            try:
                df = pd.read_csv(csv_file)
                print(f"   Loaded {len(df):,} entries")
                
                # Assuming CSV has columns like 'filename' and 'emotion' or 'label'
                for _, row in df.iterrows():
                    try:
                        # Try different possible column names
                        if 'filename' in df.columns:
                            img_name = row['filename']
                        elif 'image' in df.columns:
                            img_name = row['image']  
                        else:
                            # Use first column as filename
                            img_name = row.iloc[0]
                        
                        # Get emotion label
                        if 'emotion' in df.columns:
                            emotion = str(row['emotion'])
                        elif 'label' in df.columns:
                            emotion = str(row['label'])
                        else:
                            # Use second column as emotion
                            emotion = str(row.iloc[1])
                        
                        # Normalize emotion
                        emotion = self.emotion_mapping.get(emotion.lower(), emotion.lower())
                        
                        # Find the actual image file
                        img_file = None
                        for possible_dir in [csv_dir / 'DATASET' / split_name, csv_dir / 'images', csv_dir]:
                            for ext in ['.jpg', '.png', '.jpeg']:
                                potential_file = possible_dir / f"{img_name}{ext}"
                                if potential_file.exists():
                                    img_file = potential_file
                                    break
                            if img_file:
                                break
                        
                        if img_file and img_file.exists():
                            # Create destination directory
                            dest_dir = self.unified_dir / split_name / emotion
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Copy image
                            dest_file = dest_dir / f"csv_{img_file.name}"
                            shutil.copy2(img_file, dest_file)
                            converted_count += 1
                    
                    except Exception as e:
                        print(f"   ⚠️ Error processing row: {e}")
                        continue
            
            except Exception as e:
                print(f"   ❌ Error reading {csv_name}: {e}")
        
        print(f"   ✅ Converted {converted_count:,} images from CSV format")
        return converted_count
    
    def copy_directory_formats(self):
        """Copy directory-structured datasets (Archive 1 and 3)."""
        print("\n📁 COPYING DIRECTORY FORMATS...")
        
        copied_count = 0
        
        # Archive 1 (FER2013)
        archive1_dir = self.base_dir / 'archive'
        if archive1_dir.exists():
            print("   Processing Archive 1 (FER2013)...")
            
            for split in ['train', 'test']:
                split_dir = archive1_dir / split
                if split_dir.exists():
                    for emotion_dir in split_dir.iterdir():
                        if emotion_dir.is_dir():
                            emotion_name = self.emotion_mapping.get(emotion_dir.name.lower(), emotion_dir.name.lower())
                            
                            # Create destination
                            dest_dir = self.unified_dir / split / emotion_name  
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Copy images
                            for img_file in emotion_dir.glob("*.jpg"):
                                dest_file = dest_dir / f"fer2013_{img_file.name}"
                                shutil.copy2(img_file, dest_file)
                                copied_count += 1
        
        # Archive 3
        archive3_dir = self.base_dir / 'archive (3)'
        if archive3_dir.exists():
            print("   Processing Archive 3...")
            
            for split in ['train', 'test', 'validation']:
                split_name = 'val' if split == 'validation' else split
                split_dir = archive3_dir / split
                
                if split_dir.exists():
                    for emotion_dir in split_dir.iterdir():
                        if emotion_dir.is_dir():
                            emotion_name = self.emotion_mapping.get(emotion_dir.name.lower(), emotion_dir.name.lower())
                            
                            # Create destination
                            dest_dir = self.unified_dir / split_name / emotion_name
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Copy images
                            for img_file in emotion_dir.glob("*.png"):
                                dest_file = dest_dir / f"arch3_{img_file.name}"
                                shutil.copy2(img_file, dest_file)
                                copied_count += 1
        
        print(f"   ✅ Copied {copied_count:,} images from directory formats")
        return copied_count
    
    def create_unified_dataset(self):
        """Create unified dataset from all sources."""
        print("\n🎯 CREATING UNIFIED DATASET")
        print("=" * 60)
        
        # Clean existing unified dataset
        if self.unified_dir.exists():
            print("🗑️ Cleaning existing unified dataset...")
            shutil.rmtree(self.unified_dir)
        
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        
        total_converted = 0
        
        # Convert each format
        total_converted += self.convert_yolo_format()
        total_converted += self.convert_csv_format() 
        total_converted += self.copy_directory_formats()
        
        # Generate statistics
        self.generate_dataset_stats()
        
        print(f"\n🎉 UNIFIED DATASET CREATED!")
        print(f"📊 Total images: {total_converted:,}")
        print(f"📁 Location: {self.unified_dir}")
        
        return total_converted
    
    def generate_dataset_stats(self):
        """Generate statistics for the unified dataset."""
        print("\n📊 GENERATING DATASET STATISTICS...")
        
        stats = {'splits': {}, 'emotions': Counter(), 'total': 0}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.unified_dir / split
            if split_dir.exists():
                split_stats = Counter()
                split_total = 0
                
                for emotion_dir in split_dir.iterdir():
                    if emotion_dir.is_dir():
                        emotion_count = len(list(emotion_dir.glob("*.jpg"))) + len(list(emotion_dir.glob("*.png")))
                        split_stats[emotion_dir.name] = emotion_count
                        split_total += emotion_count
                        stats['emotions'][emotion_dir.name] += emotion_count
                
                stats['splits'][split] = dict(split_stats)
                stats['splits'][f'{split}_total'] = split_total
                stats['total'] += split_total
        
        # Save stats
        with open(self.unified_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print(f"\n📈 DATASET SUMMARY:")
        for split, split_stats in stats['splits'].items():
            if not split.endswith('_total'):
                total = stats['splits'].get(f'{split}_total', 0)
                print(f"   {split.upper()}: {total:,} images")
                
        print(f"\n🎭 EMOTION DISTRIBUTION:")
        for emotion, count in stats['emotions'].most_common():
            print(f"   {emotion:>10}: {count:6,} images")
        
        print(f"\n🎯 TOTAL: {stats['total']:,} images")

def main():
    """Main function."""
    print("🎯 DATASET ORGANIZATION AND CONVERSION")
    print("=" * 80)
    
    organizer = DatasetOrganizer()
    
    # Analyze current datasets
    total_images = organizer.analyze_datasets()
    
    if total_images == 0:
        print("❌ No datasets found!")
        return
    
    print(f"\n📋 This script will:")
    print("1. Convert YOLO format → Directory structure")
    print("2. Convert CSV format → Directory structure") 
    print("3. Copy existing directory formats")
    print("4. Create unified dataset with consistent naming")
    print("5. Generate dataset statistics")
    
    proceed = input(f"\n🚀 Proceed with conversion? (y/n): ").lower().strip()
    
    if proceed == 'y':
        organizer.create_unified_dataset()
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Train with unified dataset:")
        print("   /Users/piyushtyagi/FacialEmotion/.venv/bin/python train.py \\")
        print("     --data_dir data/unified_dataset \\")
        print("     --dataset_type directory \\")
        print("     --num_epochs 50 \\")
        print("     --batch_size 32")
        
        print(f"\n2. Monitor training:")
        print("   tensorboard --logdir outputs/logs")
        
    else:
        print("👋 Conversion cancelled.")

if __name__ == '__main__':
    main()
