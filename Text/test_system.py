"""
Test script to verify the emotion detection system works correctly.
This script runs basic tests without requiring a full training cycle.
"""

import sys
import os
import traceback
from datetime import datetime


def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('sklearn', 'Scikit-learn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'TQDM')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All imports successful!")
        return True


def test_device_detection():
    """Test device detection and GPU availability."""
    print("\n🧪 Testing device detection...")
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  📱 Device: {device}")
        
        if torch.cuda.is_available():
            print(f"  🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  💻 Using CPU (GPU not available)")
        
        print("✅ Device detection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Device detection failed: {str(e)}")
        return False


def test_tokenizer_loading():
    """Test if tokenizer can be loaded."""
    print("\n🧪 Testing tokenizer loading...")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Test tokenization
        sample_text = "I am happy today!"
        tokens = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"  📝 Sample text: \"{sample_text}\"")
        print(f"  🔤 Tokens shape: {tokens['input_ids'].shape}")
        print(f"  📚 Vocab size: {tokenizer.vocab_size}")
        
        print("✅ Tokenizer loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {str(e)}")
        return False


def test_model_loading():
    """Test if model can be loaded."""
    print("\n🧪 Testing model loading...")
    
    try:
        from transformers import AutoModelForSequenceClassification
        import torch
        
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=6  # 6 emotions
        )
        
        print(f"  🤖 Model type: {model.config.model_type}")
        print(f"  📊 Parameters: {model.num_parameters():,}")
        print(f"  🎯 Num labels: {model.config.num_labels}")
        
        # Test forward pass
        sample_input = torch.randint(0, 1000, (1, 10))  # Random token IDs
        attention_mask = torch.ones_like(sample_input)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=sample_input, attention_mask=attention_mask)
            logits = outputs.logits
        
        print(f"  📈 Output shape: {logits.shape}")
        print("✅ Model loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False


def test_dataset_access():
    """Test if dataset can be accessed."""
    print("\n🧪 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Load just a small sample to test connection
        dataset = load_dataset("dair-ai/emotion", split="train[:10]")
        
        print(f"  📊 Sample size: {len(dataset)}")
        print(f"  📝 Features: {list(dataset.features.keys())}")
        
        # Show sample
        sample = dataset[0]
        print(f"  📄 Sample text: \"{sample['text'][:50]}...\"")
        print(f"  🏷️  Sample label: {sample['label']}")
        
        print("✅ Dataset access successful!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset access failed: {str(e)}")
        print("  💡 This may be due to internet connection or HuggingFace API limits")
        return False


def test_emotion_classifier_init():
    """Test if EmotionClassifier can be initialized."""
    print("\n🧪 Testing EmotionClassifier initialization...")
    
    try:
        # Import our custom class
        from emotion_detection import EmotionClassifier
        
        classifier = EmotionClassifier(
            model_name="bert-base-uncased",
            max_length=128
        )
        
        print(f"  🤖 Model name: {classifier.model_name}")
        print(f"  📏 Max length: {classifier.max_length}")
        print(f"  📱 Device: {classifier.device}")
        print(f"  🎭 Emotions: {list(classifier.emotion_labels.values())}")
        
        print("✅ EmotionClassifier initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ EmotionClassifier initialization failed: {str(e)}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test if configuration can be loaded."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from config import MODEL_CONFIGS, DATASET_CONFIG, TRAINING_CONFIG
        
        print(f"  🤖 Available models: {list(MODEL_CONFIGS.keys())}")
        print(f"  🎭 Emotions: {list(DATASET_CONFIG['emotion_labels'].values())}")
        print(f"  ⚙️  Training output dir: {TRAINING_CONFIG['output_dir']}")
        
        print("✅ Configuration loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {str(e)}")
        return False


def test_utils_loading():
    """Test if utility functions can be loaded."""
    print("\n🧪 Testing utilities loading...")
    
    try:
        from utils import create_directories, validate_text_input, format_prediction_output
        
        # Test create_directories
        test_dirs = ["./test_dir1", "./test_dir2"]
        create_directories(test_dirs)
        
        # Test validate_text_input
        test_text = "  This is a test!  "
        cleaned = validate_text_input(test_text)
        print(f"  📝 Text validation: \"{test_text}\" -> \"{cleaned}\"")
        
        # Test format_prediction_output
        mock_result = {
            'text': 'I am happy!',
            'predicted_emotion': 'joy',
            'confidence': 0.95,
            'top_predictions': [('joy', 0.95), ('love', 0.03), ('surprise', 0.02)]
        }
        formatted = format_prediction_output(mock_result)
        print(f"  📊 Formatted output length: {len(formatted)} chars")
        
        # Clean up test directories
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
        
        print("✅ Utilities loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Utilities loading failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 EMOTION DETECTION - SYSTEM TESTS")
    print("=" * 60)
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Package Imports", test_imports),
        ("Device Detection", test_device_detection),
        ("Tokenizer Loading", test_tokenizer_loading),
        ("Model Loading", test_model_loading),
        ("Dataset Access", test_dataset_access),
        ("EmotionClassifier Init", test_emotion_classifier_init),
        ("Configuration Loading", test_config_loading),
        ("Utilities Loading", test_utils_loading)
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
                results.append((test_name, "✅ PASS"))
            else:
                failed += 1
                results.append((test_name, "❌ FAIL"))
        except Exception as e:
            failed += 1
            results.append((test_name, f"❌ ERROR: {str(e)}"))
            print(f"❌ Unexpected error in {test_name}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        print(f"{result:<12} {test_name}")
    
    print(f"\n📈 Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        print("💡 Next steps:")
        print("   1. Run 'python emotion_detection.py' to train a model")
        print("   2. Run 'python examples.py' to see usage examples")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        print("💡 Troubleshooting:")
        print("   1. Run 'python setup.py' to install dependencies")
        print("   2. Check internet connection for dataset/model downloads")
        print("   3. Ensure Python 3.7+ is installed")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
