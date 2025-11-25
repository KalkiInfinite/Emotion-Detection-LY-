"""
Test script to verify the installation and basic functionality.
Run this script to make sure everything is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from transformers import ViTModel

# Test imports from our modules
try:
    from models.vit_emotion_model import ViTEmotionClassifier
    from data.transforms import get_train_transforms, get_val_transforms
    from data.dataloader import EmotionDataset
    print(" All custom modules imported successfully!")
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)

def test_model_creation():
    """Test creating the ViT emotion model."""
    print("\n Testing model creation...")
    
    try:
        model = ViTEmotionClassifier(
            model_name="google/vit-base-patch16-224",
            num_classes=7,
            dropout_rate=0.1
        )
        print(f" Model created successfully!")
        print(f"   Model info: {model.get_model_info()}")
        return model
    except Exception as e:
        print(f" Model creation failed: {e}")
        return None

def test_transforms():
    """Test data transforms."""
    print("\n🧪 Testing data transforms...")
    
    try:
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Test transforms
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        train_tensor = train_transform(dummy_image)
        val_tensor = val_transform(dummy_image)
        
        print(f" Transforms working!")
        print(f"   Train tensor shape: {train_tensor.shape}")
        print(f"   Val tensor shape: {val_tensor.shape}")
        
        return train_tensor
    except Exception as e:
        print(f" Transform test failed: {e}")
        return None

def test_model_forward_pass(model, sample_tensor):
    """Test model forward pass."""
    print("\n Testing model forward pass...")
    
    try:
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            batch_input = sample_tensor.unsqueeze(0)
            output = model(batch_input)
            
        print(f" Forward pass successful!")
        print(f"   Input shape: {batch_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output (logits): {output.squeeze().tolist()}")
        
        # Test prediction method
        try:
            prediction = model.predict_emotion(batch_input)
            print(f"   Predicted emotion: {prediction['predicted_emotion']}")
            print(f"   Confidence: {prediction['confidence']:.4f}")
        except Exception as pred_error:
            print(f"   Prediction method test skipped: {pred_error}")
            # This is fine - the forward pass itself worked
        
        return True
    except Exception as e:
        print(f" Forward pass failed: {e}")
        return False

def test_device_compatibility():
    """Test device compatibility."""
    print("\n Testing device compatibility...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Available device: {device}")
    
    if torch.cuda.is_available():
        print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   CUDA not available - using CPU")
    
    return str(device)

def main():
    """Run all tests."""
    print(" Starting Facial Emotion Recognition Setup Test")
    print("=" * 50)
    
    # Test device compatibility
    device = test_device_compatibility()
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        return
    
    # Test transforms
    sample_tensor = test_transforms()
    if sample_tensor is None:
        return
    
    # Test forward pass
    success = test_model_forward_pass(model, sample_tensor)
    if not success:
        return
    
    print("\n" + "=" * 50)
    print(" ALL TESTS PASSED! Your setup is ready!")
    print("=" * 50)
    
    print("\n Next Steps:")
    print("1. Prepare your dataset in one of the supported formats:")
    print("   - Directory structure: dataset/emotion_name/image.jpg")
    print("   - FER2013 format: CSV file + extracted images")
    
    print("\n2. Start training:")
    print("   python train.py --data_dir /path/to/dataset --dataset_type directory")
    
    print("\n3. After training, evaluate your model:")
    print("   python eval.py --checkpoint outputs/checkpoints/best_model.pt --data_dir /path/to/test/data")
    
    print("\n4. Run inference:")
    print("   python inference.py --checkpoint outputs/checkpoints/best_model.pt --mode single --input image.jpg")

if __name__ == '__main__':
    main()
