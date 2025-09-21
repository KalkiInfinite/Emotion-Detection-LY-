"""
Example usage script demonstrating different ways to use the emotion detection model.
"""

from emotion_detection import EmotionClassifier
from config import MODEL_CONFIGS
from utils import format_prediction_output, validate_text_input
import os


def quick_prediction_example():
    """Quick example for predicting emotions on sample texts."""
    print("🚀 QUICK PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Sample texts representing different emotions
    sample_texts = [
        "I'm absolutely thrilled! This is the best day of my life!",
        "I'm terrified of what might happen in the future.",
        "You mean everything to me. I love you so much.",
        "This is so frustrating! Why does this always happen to me?",
        "I feel so empty and lost. Nothing seems to matter anymore.",
        "Wow! I never saw that coming! What an incredible surprise!"
    ]
    
    try:
        # Initialize classifier and load your trained model
        classifier = EmotionClassifier(model_name="bert-base-uncased")
        
        # Load your trained model
        classifier.load_saved_model("./saved_emotion_model")
        
        # Make predictions
        results = classifier.predict_emotion(sample_texts)
        
        # Display formatted results
        for result in results:
            print(format_prediction_output(result))
            
    except Exception as e:
        print(f"❌ Error in quick prediction: {str(e)}")


def interactive_prediction():
    """Interactive prediction where user can input custom text."""
    print("\n🎭 INTERACTIVE EMOTION PREDICTION")
    print("=" * 50)
    print("Enter text to analyze emotions (type 'quit' to exit):")
    
    try:
        # Load pre-trained model
        classifier = EmotionClassifier()
        classifier.load_saved_model("./saved_emotion_model")
        
        while True:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️ Please enter some text.")
                continue
            
            try:
                # Validate and predict
                clean_text = validate_text_input(text)
                result = classifier.predict_emotion([clean_text])[0]
                print(format_prediction_output(result))
                
            except Exception as e:
                print(f"❌ Error predicting emotion: {str(e)}")
                
    except Exception as e:
        print(f"❌ Error in interactive mode: {str(e)}")
        print("💡 Make sure you have a trained model saved in './saved_emotion_model'")


def batch_prediction_example():
    """Example of batch prediction from a file."""
    print("\n📄 BATCH PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Create sample data file
    sample_file = "sample_texts.txt"
    sample_texts = [
        "The concert was amazing! I had such a wonderful time.",
        "I'm worried about the results of my medical tests.",
        "Being with you makes me feel complete and happy.",
        "The traffic is terrible and I'm going to be late again!",
        "I miss my grandmother so much. She was everything to me.",
        "I can't believe I won the lottery! This is incredible!"
    ]
    
    try:
        # Write sample texts to file
        with open(sample_file, 'w') as f:
            for text in sample_texts:
                f.write(text + '\n')
        
        print(f"📝 Created sample file: {sample_file}")
        
        # Load texts from file
        with open(sample_file, 'r') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Initialize and load model
        classifier = EmotionClassifier()
        classifier.load_saved_model("./saved_emotion_model")
        
        # Batch prediction
        print(f"\n🔮 Predicting emotions for {len(texts)} texts...")
        results = classifier.predict_emotion(texts)
        
        # Save results to file
        results_file = "batch_prediction_results.txt"
        with open(results_file, 'w') as f:
            f.write("BATCH EMOTION PREDICTION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Text {i}: {result['text']}\n")
                f.write(f"Predicted Emotion: {result['predicted_emotion']}\n")
                f.write(f"Confidence: {result['confidence']:.3f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"💾 Results saved to: {results_file}")
        
        # Clean up
        os.remove(sample_file)
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")


def model_comparison_example():
    """Example comparing different model architectures."""
    print("\n🏆 MODEL COMPARISON EXAMPLE")
    print("=" * 50)
    
    models_to_compare = ['bert-base', 'roberta-base', 'distilbert']
    test_text = "I'm so excited about this new opportunity! It's going to be amazing!"
    
    print(f"📝 Test text: \"{test_text}\"")
    print("\n🤖 Comparing different models:")
    
    for model_key in models_to_compare:
        try:
            config = MODEL_CONFIGS[model_key]
            print(f"\n--- {config['model_name']} ---")
            
            classifier = EmotionClassifier(
                model_name=config['model_name'],
                max_length=config['max_length']
            )
            
            # Load model (assuming they're pre-trained and saved)
            model_path = f"./saved_{model_key}_model"
            if os.path.exists(model_path):
                classifier.load_saved_model(model_path)
                result = classifier.predict_emotion([test_text])[0]
                
                print(f"Predicted: {result['predicted_emotion']} ({result['confidence']:.3f})")
                print("Top 3:")
                for emotion, prob in result['top_predictions']:
                    print(f"  {emotion}: {prob:.3f}")
            else:
                print(f"⚠️ Model not found at {model_path}")
                
        except Exception as e:
            print(f"❌ Error with {model_key}: {str(e)}")


def evaluation_metrics_example():
    """Example showing how to evaluate model performance."""
    print("\n📊 EVALUATION METRICS EXAMPLE")
    print("=" * 50)
    
    try:
        # This would typically be done after training
        # Here we show how to load and evaluate a saved model
        
        classifier = EmotionClassifier()
        classifier.load_saved_model("./saved_emotion_model")
        
        # Load dataset for evaluation
        dataset = classifier.load_dataset()
        tokenized_dataset = classifier.preprocess_dataset(dataset)
        
        # Create a trainer for evaluation (without training)
        classifier.setup_training(tokenized_dataset)
        
        # Evaluate on test set
        eval_results, y_true, y_pred = classifier.evaluate_model(tokenized_dataset)
        
        print("\n📈 Performance Summary:")
        print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"Test Loss: {eval_results['eval_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in evaluation: {str(e)}")
        print("💡 Make sure you have a trained model and internet connection for dataset loading")


def main():
    """Main function to run all examples."""
    print("🎭 EMOTION DETECTION - USAGE EXAMPLES")
    print("=" * 60)
    
    print("\nAvailable examples:")
    print("1. Quick Prediction Example")
    print("2. Interactive Prediction")
    print("3. Batch Prediction Example")
    print("4. Model Comparison Example")
    print("5. Evaluation Metrics Example")
    print("6. Run All Examples")
    
    try:
        choice = input("\nSelect an example (1-6): ").strip()
        
        if choice == '1':
            quick_prediction_example()
        elif choice == '2':
            interactive_prediction()
        elif choice == '3':
            batch_prediction_example()
        elif choice == '4':
            model_comparison_example()
        elif choice == '5':
            evaluation_metrics_example()
        elif choice == '6':
            print("\n🚀 Running all examples...")
            quick_prediction_example()
            batch_prediction_example()
            model_comparison_example()
            evaluation_metrics_example()
            # Skip interactive mode in "run all"
        else:
            print("❌ Invalid choice. Please select 1-6.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
