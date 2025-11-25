"""
Ultra-Flexible Training - Start with ANY number of epochs, stop anytime, resume with different targets
"""

import os
import sys
import json
from pathlib import Path

import os
import sys
import argparse
import time
from datetime import datetime

def check_checkpoint_status(checkpoint_dir="./outputs/safe_training"):
    """Check the current training status."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    history_path = Path(checkpoint_dir) / "training_history.json"
    
    print("🔍 TRAINING STATUS CHECK")
    print("=" * 50)
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found. Starting fresh.")
        return False, {}
    
    # Load checkpoint info
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📂 Checkpoint found!")
        print(f"   Current epoch: {checkpoint['epoch'] + 1}")
        print(f"   Current batch: {checkpoint['batch'] + 1} / {checkpoint.get('total_batches_per_epoch', '?')}")
        print(f"   Total batches trained so far: {checkpoint.get('total_batches_trained', 0):,}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        print(f"   Last training session target: {checkpoint.get('target_epochs', 'unknown')} epochs")
        
        # Calculate progress
        current_epoch = checkpoint['epoch'] + 1
        current_batch = checkpoint['batch'] + 1
        
        # Load history if available
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                print(f"   Training history entries: {len(history.get('training_history', []))}")
                
                # Show recent progress
                if 'training_history' in history and history['training_history']:
                    recent = history['training_history'][-3:]
                    print(f"   Recent progress:")
                    for entry in recent:
                        if 'val_accuracy' in entry:
                            print(f"     Epoch {entry['epoch']}: {entry['val_accuracy']:.2f}% accuracy")
        
        return True, {
            'current_epoch': current_epoch,
            'current_batch': current_batch,
            'best_val_acc': checkpoint.get('best_val_acc', 0),
            'total_batches_trained': checkpoint.get('total_batches_trained', 0)
        }
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False, {}
from train import EmotionTrainer  # Import the trainer class

def main():
    """Main training function for pre-split dataset."""
    parser = argparse.ArgumentParser(description='Train ViT on Pre-Split Dataset')
    parser.add_argument('--data_dir', type=str, default='data/unified_dataset', 
                       help='Path to dataset directory with train/val/test subdirs')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224',
                       help='Pretrained ViT model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./outputs/unified_model', 
                       help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    train_transform = get_train_transforms(use_augmentation=True)
    val_transform = get_val_transforms()
    
    # Load pre-split datasets
    print("Loading datasets...")
    train_dataset = DirectoryDataset.from_directory(
        os.path.join(args.data_dir, 'train'), 
        transform=train_transform
    )
    val_dataset = DirectoryDataset.from_directory(
        os.path.join(args.data_dir, 'val'), 
        transform=val_transform
    )
    
    print(f"Train dataset: {len(train_dataset):,} images")
    print(f"Val dataset: {len(val_dataset):,} images") 
    print(f"Emotions: {train_dataset.emotion_labels}")
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampling=True
    )
    
    # Model
    num_classes = len(train_dataset.emotion_labels)
    model = ViTEmotionClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        dropout_rate=0.1,
        freeze_backbone=False
    ).to(device)
    
    print(f'Model info: {model.get_model_info()}')
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        log_dir=f"{args.save_dir}/logs",
        emotion_labels=train_dataset.emotion_labels
    )
    
    # Train
    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train(num_epochs=args.num_epochs)
    
    print("🎉 Training completed!")
    print(f"📁 Models saved in: {args.save_dir}")
    print(f"📊 Best validation accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
