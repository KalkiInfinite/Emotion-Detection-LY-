"""
Flexible Training Options - Choose your training duration and expected results
"""

import os
import sys
import json
from pathlib import Path

def check_checkpoint_status(checkpoint_dir="./outputs/safe_training"):
    """Check the current training status."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    
    print("🔍 TRAINING STATUS CHECK")
    print("=" * 50)
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found. Starting fresh.")
        return False
    
    # Load checkpoint info
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📂 Checkpoint found!")
        print(f"   Current epoch: {checkpoint['epoch'] + 1}")
        print(f"   Current batch: {checkpoint['batch'] + 1}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def quick_training(epochs=15):
    """Start quick training with fewer epochs."""
    print(f"\n🚀 QUICK TRAINING ({epochs} epochs)")
    print("=" * 50)
    print(f"⏱️  Estimated time: 6-14 hours")
    print(f"🎯 Expected accuracy: 75-85%")
    print(f"💡 Perfect for testing and quick results!")
    
    cmd = f"""
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \\
    --data_dir data/unified_dataset \\
    --dataset_type pre_split \\
    --batch_size 32 \\
    --max_epochs {epochs} \\
    --learning_rate 1e-4 \\
    --validate_every 300
"""
    
    print("🚀 Starting quick training...")
    print("💡 Remember: Type 'stop' or 'quit' and press Enter to save and exit safely")
    print(f"\nCommand being executed:")
    print(cmd.strip())
    
    os.system(cmd)

def full_training(epochs=50):
    """Start full training with all epochs."""
    print(f"\n🎯 FULL TRAINING ({epochs} epochs)")
    print("=" * 50)
    print(f"⏱️  Estimated time: 22-43 hours")
    print(f"🎯 Expected accuracy: 85-92%")
    print(f"🏆 Maximum performance training!")
    
    cmd = f"""
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \\
    --data_dir data/unified_dataset \\
    --dataset_type pre_split \\
    --batch_size 32 \\
    --max_epochs {epochs} \\
    --learning_rate 1e-4 \\
    --validate_every 300
"""
    
    print("🚀 Starting full training...")
    print("💡 Remember: Type 'stop' or 'quit' and press Enter to save and exit safely")
    print(f"\nCommand being executed:")
    print(cmd.strip())
    
    os.system(cmd)

def resume_training():
    """Resume from checkpoint."""
    print(f"\n🔄 RESUMING TRAINING")
    print("=" * 50)
    
    cmd = """
cd /Users/piyushtyagi/FacialEmotion && \\
/Users/piyushtyagi/FacialEmotion/.venv/bin/python train_safe.py \\
    --data_dir data/unified_dataset \\
    --dataset_type pre_split \\
    --batch_size 32 \\
    --max_epochs 50 \\
    --learning_rate 1e-4 \\
    --validate_every 300 \\
    --resume
"""
    
    print("🚀 Resuming training...")
    print("💡 Remember: Type 'stop' or 'quit' and press Enter to save and exit safely")
    print("\nCommand being executed:")
    print(cmd.strip())
    
    os.system(cmd)

def main():
    """Main function."""
    print("⚡ FLEXIBLE TRAINING OPTIONS")
    print("=" * 60)
    
    # Check if checkpoint exists
    has_checkpoint = check_checkpoint_status()
    
    print(f"\n📊 TRAINING TIME & ACCURACY GUIDE:")
    print("   • 10-15 epochs: 4-14 hours → 70-80% accuracy")
    print("   • 20-30 epochs: 9-26 hours → 80-87% accuracy") 
    print("   • 40-50 epochs: 18-43 hours → 85-92% accuracy")
    
    print(f"\n📋 AVAILABLE OPTIONS:")
    
    if has_checkpoint:
        print("1. Resume training from checkpoint")
        print("2. Quick training - 15 epochs (~6-14 hours, 75-85% accuracy)")
        print("3. Full training - 50 epochs (~22-43 hours, 85-92% accuracy)")
        print("4. Custom epochs (5-100)")
    else:
        print("1. Quick training - 15 epochs (~6-14 hours, 75-85% accuracy)")
        print("2. Full training - 50 epochs (~22-43 hours, 85-92% accuracy)")
        print("3. Custom epochs (5-100)")
    
    print("0. Exit")
    
    try:
        choice = input(f"\nEnter your choice: ").strip()
        
        if has_checkpoint:
            if choice == '1':
                resume_training()
            elif choice == '2':
                confirm = input("⚠️ This will start fresh training (checkpoint will be overwritten). Continue? (y/N): ").strip().lower()
                if confirm == 'y':
                    quick_training(15)
                else:
                    print("❌ Cancelled")
            elif choice == '3':
                confirm = input("⚠️ This will start fresh training (checkpoint will be overwritten). Continue? (y/N): ").strip().lower()
                if confirm == 'y':
                    full_training(50)
                else:
                    print("❌ Cancelled")
            elif choice == '4':
                try:
                    epochs = int(input("Enter number of epochs (5-100): "))
                    if 5 <= epochs <= 100:
                        confirm = input(f"⚠️ Start fresh training with {epochs} epochs? (y/N): ").strip().lower()
                        if confirm == 'y':
                            if epochs <= 20:
                                quick_training(epochs)
                            else:
                                full_training(epochs)
                        else:
                            print("❌ Cancelled")
                    else:
                        print("❌ Invalid number of epochs (must be 5-100)")
                except ValueError:
                    print("❌ Invalid input - please enter a number")
            elif choice == '0':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
        else:
            if choice == '1':
                quick_training(15)
            elif choice == '2':
                full_training(50)
            elif choice == '3':
                try:
                    epochs = int(input("Enter number of epochs (5-100): "))
                    if 5 <= epochs <= 100:
                        if epochs <= 20:
                            quick_training(epochs)
                        else:
                            full_training(epochs)
                    else:
                        print("❌ Invalid number of epochs (must be 5-100)")
                except ValueError:
                    print("❌ Invalid input - please enter a number")
            elif choice == '0':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
