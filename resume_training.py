"""
Resume Training Script - Continue from where you left off
"""

import os
import sys
import json
from pathlib import Path

def check_checkpoint_status(checkpoint_dir="./outputs/safe_training"):
    """Check the current training status."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    history_path = Path(checkpoint_dir) / "training_history.json"
    
    print("🔍 TRAINING STATUS CHECK")
    print("=" * 50)
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found. Start fresh training.")
        return False
    
    # Load checkpoint info
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📂 Checkpoint found!")
        print(f"   Current epoch: {checkpoint['epoch'] + 1}")
        print(f"   Current batch: {checkpoint['batch'] + 1}")
        print(f"   Total batches trained: {checkpoint.get('total_batches_trained', 0):,}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        print(f"   Device used: {checkpoint.get('device_used', 'unknown')}")
        print(f"   Last saved: {checkpoint.get('timestamp', 'unknown')}")
        
        # Load history if available
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                print(f"   Training history entries: {len(history.get('training_history', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def resume_training():
    """Resume training from checkpoint."""
    print("\n🔄 RESUMING TRAINING")
    print("=" * 50)
    
    # Run the safe training script
    cmd = """
cd /Users/piyushtyagi/FacialEmotion && \
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \
    --data_dir data/unified_dataset \
    --dataset_type pre_split \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 1e-4 \
    --validate_every 300 \
    --resume
"""
    
    print("🚀 Starting resumed training...")
    print("💡 Remember: Type 'stop' or 'quit' and press Enter to save and exit safely")
    print("\nCommand being executed:")
    print(cmd.strip())
    
    os.system(cmd)

def start_fresh():
    """Start fresh training."""
    print("\n🆕 STARTING FRESH TRAINING")
    print("=" * 50)
    
    cmd = """
cd /Users/piyushtyagi/FacialEmotion && \
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \
    --data_dir data/unified_dataset \
    --dataset_type pre_split \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 1e-4 \
    --validate_every 300
"""
    
    print("🚀 Starting fresh training...")
    print("💡 Remember: Type 'stop' or 'quit' and press Enter to save and exit safely")
    print("\nCommand being executed:")
    print(cmd.strip())
    
    os.system(cmd)

def main():
    """Main function."""
    print("🎯 SAFE ViT EMOTION TRAINING MANAGER")
    print("=" * 60)
    
    # Check if checkpoint exists
    has_checkpoint = check_checkpoint_status()
    
    print(f"\n📋 OPTIONS:")
    if has_checkpoint:
        print("1. Resume training from checkpoint")
        print("2. Start fresh training (will overwrite checkpoint)")
    else:
        print("1. Start fresh training")
    
    print("3. Just check status and exit")
    
    try:
        choice = input(f"\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            if has_checkpoint:
                resume_training()
            else:
                start_fresh()
        elif choice == '2' and has_checkpoint:
            confirm = input("⚠️ This will overwrite existing checkpoint. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                start_fresh()
            else:
                print("❌ Cancelled")
        elif choice == '3':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
