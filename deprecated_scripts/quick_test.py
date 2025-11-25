#!/usr/bin/env python3
"""
Quick performance comparison script.
"""

import torch
import time
import os

def check_device_performance():
    """Test MPS vs CPU performance."""
    print("🧪 PERFORMANCE TEST")
    print("=" * 50)
    
    # Test data
    batch_size = 32
    image_size = (3, 224, 224)
    num_classes = 8
    
    # Create test tensors
    test_images = torch.randn(batch_size, *image_size)
    test_labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test MPS
    if torch.backends.mps.is_available():
        device_mps = torch.device('mps')
        test_images_mps = test_images.to(device_mps)
        test_labels_mps = test_labels.to(device_mps)
        
        # Simple model for testing
        model_mps = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*224*224, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        ).to(device_mps)
        
        # Time MPS
        start_time = time.time()
        for i in range(100):
            output = model_mps(test_images_mps)
            loss = torch.nn.functional.cross_entropy(output, test_labels_mps)
            loss.backward()
        torch.mps.synchronize()  # Wait for MPS operations
        mps_time = time.time() - start_time
        
        print(f"🚀 MPS (100 iterations): {mps_time:.2f} seconds")
    else:
        print("❌ MPS not available")
        mps_time = float('inf')
    
    # Test CPU
    device_cpu = torch.device('cpu')
    test_images_cpu = test_images.to(device_cpu)
    test_labels_cpu = test_labels.to(device_cpu)
    
    model_cpu = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3*224*224, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, num_classes)
    ).to(device_cpu)
    
    # Time CPU
    start_time = time.time()
    for i in range(100):
        output = model_cpu(test_images_cpu)
        loss = torch.nn.functional.cross_entropy(output, test_labels_cpu)
        loss.backward()
    cpu_time = time.time() - start_time
    
    print(f"🐌 CPU (100 iterations): {cpu_time:.2f} seconds")
    
    if mps_time != float('inf'):
        speedup = cpu_time / mps_time
        print(f"⚡ MPS is {speedup:.1f}x faster than CPU")
    
    print("=" * 50)

def show_dataset_info():
    """Show dataset information."""
    data_dir = './data/unified_dataset'
    
    if os.path.exists(data_dir):
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        if os.path.exists(train_dir):
            train_count = sum([len(files) for r, d, files in os.walk(train_dir)])
            print(f"📊 Training images: {train_count:,}")
        
        if os.path.exists(val_dir):
            val_count = sum([len(files) for r, d, files in os.walk(val_dir)])
            print(f"📊 Validation images: {val_count:,}")
    else:
        print("❌ Dataset not found at ./data/unified_dataset")

if __name__ == '__main__':
    check_device_performance()
    show_dataset_info()
