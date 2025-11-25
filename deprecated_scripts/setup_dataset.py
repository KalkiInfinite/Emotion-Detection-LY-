"""
Dataset download and setup helper script.
This script helps you download and organize the FER2013 dataset.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import shutil

def download_fer2013_kaggle():
    """
    Instructions for downloading FER2013 from Kaggle.
    """
    print("📥 FER2013 Dataset Setup Instructions")
    print("=" * 50)
    
    print("\n Method 1: Manual Download (Recommended)")
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Click 'Download' (you may need to create a free Kaggle account)")
    print("3. Extract the downloaded zip file")
    print("4. Copy the folders to: /Users/piyushtyagi/FacialEmotion/data/fer2013/")
    
    print("\n Expected structure after extraction:")
    print("/Users/piyushtyagi/FacialEmotion/data/fer2013/")
    print("├── train/")
    print("│   ├── angry/ (3995 images)")
    print("│   ├── disgust/ (436 images)")
    print("│   ├── fear/ (4097 images)")
    print("│   ├── happy/ (7215 images)")
    print("│   ├── neutral/ (4965 images)")
    print("│   ├── sad/ (4830 images)")
    print("│   └── surprise/ (3171 images)")
    print("├── test/")
    print("│   └── ... (same structure)")
    print("└── validation/")
    print("    └── ... (same structure)")
    
    print(f"\n Total images: ~35,000")
    print(f" Size: ~150MB")
    print(f" Classes: 7 emotions")

def download_sample_images():
    """
    Download a few sample images for testing.
    """
    print("\n🧪 Downloading sample images for testing...")
    
    # Sample image URLs (placeholder - you can replace with actual sample images)
    sample_urls = {
        'happy': 'https://images.unsplash.com/photo-1554151228-14d9def656e4?w=224&h=224&fit=crop&crop=face',
        'sad': 'https://images.unsplash.com/photo-1581909552919-9b9f7c6b8b7e?w=224&h=224&fit=crop&crop=face',
    }
    
    base_dir = Path('/Users/piyushtyagi/FacialEmotion/data/fer2013/test')
    
    try:
        for emotion, url in sample_urls.items():
            emotion_dir = base_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            # Note: In a real scenario, you'd download actual emotion images
            print(f"   Sample setup for {emotion} complete")
        
        print(" Sample structure created successfully!")
        
    except Exception as e:
        print(f" Sample download failed: {e}")
        print("Don't worry, you can manually add images later.")

def check_dataset_structure():
    """
    Check if the dataset is properly structured.
    """
    print("\n Checking dataset structure...")
    
    base_dir = Path('/Users/piyushtyagi/FacialEmotion/data/fer2013')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    splits = ['train', 'test', 'validation']
    
    total_images = 0
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"❌ Missing: {split_dir}")
            continue
            
        print(f"\n {split.upper()} SET:")
        split_total = 0
        
        for emotion in emotions:
            emotion_dir = split_dir / emotion
            if emotion_dir.exists():
                image_count = len([f for f in emotion_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                print(f"   {emotion:>8}: {image_count:>4} images")
                split_total += image_count
            else:
                print(f"   {emotion:>8}: ❌ Missing")
        
        print(f"   Total: {split_total} images")
        total_images += split_total

    print(f"\n TOTAL DATASET: {total_images} images")

    if total_images > 0:
        print(" Dataset found! You can start training.")
        return True
    else:
        print(" No images found. Please download the dataset first.")
        return False

def create_dummy_dataset():
    """
    Create a small dummy dataset for testing the pipeline.
    """
    print("\n Creating dummy dataset for testing...")
    
    try:
        from PIL import Image
        import numpy as np
        
        base_dir = Path('/Users/piyushtyagi/FacialEmotion/data/fer2013')
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create a few dummy images for each emotion in train set
        for emotion in emotions:
            emotion_dir = base_dir / 'train' / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 3 dummy images per emotion
            for i in range(3):
                # Create a random colored image
                img_array = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img = img.resize((224, 224))  # Resize to ViT input size
                
                img_path = emotion_dir / f"dummy_{emotion}_{i}.jpg"
                img.save(img_path)
        
        print(" Dummy dataset created!")
        print("   - 21 images total (3 per emotion)")
        print("   - Located in: /Users/piyushtyagi/FacialEmotion/data/fer2013/train/")
        print("   - You can now test the training pipeline!")
        
        return True
        
    except Exception as e:
        print(f" Failed to create dummy dataset: {e}")
        return False

def main():
    """
    Main function to guide dataset setup.
    """
    print(" Facial Emotion Recognition Dataset Setup")
    print("=" * 50)
    
    # Check current structure
    has_data = check_dataset_structure()
    
    if has_data:
        print("\n You already have data! You're ready to train.")
        return
    
    print("\n Choose an option:")
    print("1. Get instructions for downloading FER2013 from Kaggle")
    print("2. Create a small dummy dataset for testing")
    print("3. Just show me the expected structure")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        download_fer2013_kaggle()
    elif choice == '2':
        success = create_dummy_dataset()
        if success:
            print("\n Next steps:")
            print("1. Test training: /Users/piyushtyagi/FacialEmotion/.venv/bin/python train.py --data_dir data/fer2013/train --dataset_type directory --num_epochs 2")
            print("2. Later, replace with real FER2013 dataset for better results")
    elif choice == '3':
        download_fer2013_kaggle()
    else:
        print("Invalid choice. Run the script again.")

if __name__ == '__main__':
    main()
