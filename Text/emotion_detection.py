"""
Emotion Detection using Transformer Models
==========================================

This script implements a complete pipeline for emotion detection from text using
transformer models (BERT/RoBERTa). It includes data loading, preprocessing, 
training, evaluation, and inference capabilities.

Author: AI Assistant
Date: September 2025
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Transformers and ML libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EmotionClassifier:
    """
    A comprehensive emotion classification system using transformer models.
    """
    
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        """
        Initialize the emotion classifier.
        
        Args:
            model_name (str): Name of the pretrained model to use
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Enhanced device selection for Mac M1/M2 (MPS), CUDA, or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Emotion labels mapping
        self.emotion_labels = {
            0: "sadness",
            1: "joy", 
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        
        self.label_to_id = {v: k for k, v in self.emotion_labels.items()}
        self.num_labels = len(self.emotion_labels)
        
        print(f"🚀 Initializing Emotion Classifier")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📏 Max Length: {self.max_length}")
        print(f"🎭 Emotions: {list(self.emotion_labels.values())}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load the tokenizer and model."""
        print("\n📥 Loading tokenizer and model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.emotion_labels,
                label2id=self.label_to_id
            )
            self.model.to(self.device)
            print("✅ Model and tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def load_dataset(self):
        """Load and explore the emotion dataset from HuggingFace."""
        print("\n📊 Loading emotion dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("dair-ai/emotion")
            
            print("✅ Dataset loaded successfully!")
            print(f"📈 Dataset info:")
            print(f"   - Train samples: {len(dataset['train'])}")
            print(f"   - Test samples: {len(dataset['test'])}")
            print(f"   - Validation samples: {len(dataset['validation'])}")
            
            # Show sample data
            print(f"\n📝 Sample data:")
            for i in range(3):
                sample = dataset['train'][i]
                emotion_name = self.emotion_labels[sample['label']]
                print(f"   Text: \"{sample['text']}\"")
                print(f"   Emotion: {emotion_name} (label: {sample['label']})")
                print()
            
            # Show label distribution
            self._show_label_distribution(dataset['train'])
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise
    
    def _show_label_distribution(self, dataset):
        """Show the distribution of emotion labels in the dataset."""
        labels = [sample['label'] for sample in dataset]
        
        # Count occurrences
        label_counts = {}
        for label in labels:
            emotion_name = self.emotion_labels[label]
            label_counts[emotion_name] = label_counts.get(emotion_name, 0) + 1
        
        print("🏷️  Label Distribution:")
        total_samples = len(labels)
        for emotion, count in sorted(label_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"   {emotion}: {count} ({percentage:.1f}%)")
    
    def preprocess_dataset(self, dataset):
        """
        Preprocess the dataset by tokenizing the text.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            tokenized dataset
        """
        print("\n🔄 Preprocessing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        try:
            # Tokenize the dataset
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=['text']  # Remove original text column
            )
            
            # Set format for PyTorch
            tokenized_dataset.set_format("torch")
            
            print("✅ Dataset preprocessing completed!")
            return tokenized_dataset
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            raise
    
    def setup_training(self, tokenized_dataset, output_dir="./emotion_model_results"):
        """
        Set up the training configuration and trainer.
        
        Args:
            tokenized_dataset: Preprocessed dataset
            output_dir: Directory to save model and results
        """
        print("\n⚙️  Setting up training configuration...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            push_to_hub=False,
            logging_steps=100,
            warmup_steps=500,
            save_total_limit=2,
            seed=42,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA GPU available
            # Note: MPS (Apple Silicon) doesn't support fp16 yet, so we keep CUDA check
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1': f1
            }
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        print("✅ Training setup completed!")
        print(f"📂 Output directory: {output_dir}")
        print(f"🏃 Epochs: {training_args.num_train_epochs}")
        print(f"📚 Batch size: {training_args.per_device_train_batch_size}")
        print(f"🎯 Learning rate: {training_args.learning_rate}")
    
    def train_model(self):
        """Train the emotion classification model."""
        print("\n🏋️ Starting model training...")
        print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model()
            
            print("✅ Training completed successfully!")
            print(f"📊 Training metrics:")
            print(f"   - Train Loss: {train_result.training_loss:.4f}")
            print(f"   - Train Runtime: {train_result.metrics['train_runtime']:.2f} seconds")
            print(f"   - Train Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}")
            
            return train_result
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            raise
    
    def evaluate_model(self, tokenized_dataset):
        """
        Evaluate the trained model on the test set.
        
        Args:
            tokenized_dataset: Preprocessed dataset
            
        Returns:
            evaluation results
        """
        print("\n📏 Evaluating model on test set...")
        
        try:
            # Evaluate on test set
            eval_results = self.trainer.evaluate(eval_dataset=tokenized_dataset['test'])
            
            print("✅ Evaluation completed!")
            print(f"📊 Test Results:")
            print(f"   - Test Loss: {eval_results['eval_loss']:.4f}")
            print(f"   - Test Accuracy: {eval_results['eval_accuracy']:.4f}")
            print(f"   - Test F1 Score: {eval_results['eval_f1']:.4f}")
            
            # Get detailed predictions for classification report
            predictions = self.trainer.predict(tokenized_dataset['test'])
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Print classification report
            print("\n📋 Detailed Classification Report:")
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=list(self.emotion_labels.values()),
                digits=4
            )
            print(report)
            
            # Create and save confusion matrix
            self._plot_confusion_matrix(y_true, y_pred)
            
            return eval_results, y_true, y_pred
            
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            raise
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=list(self.emotion_labels.values()),
                yticklabels=list(self.emotion_labels.values())
            )
            
            plt.title('Emotion Classification - Confusion Matrix')
            plt.xlabel('Predicted Emotion')
            plt.ylabel('True Emotion')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("📊 Confusion matrix saved as 'confusion_matrix.png'")
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Could not create confusion matrix plot: {str(e)}")
    
    def predict_emotion(self, texts):
        """
        Predict emotions for new text samples.
        
        Args:
            texts (str or list): Text(s) to classify
            
        Returns:
            predictions with probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"\n🔮 Predicting emotions for {len(texts)} text(s)...")
        
        try:
            # Tokenize the input texts
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            results = []
            for i, text in enumerate(texts):
                probs = predictions[i].cpu().numpy()
                predicted_label = np.argmax(probs)
                predicted_emotion = self.emotion_labels[predicted_label]
                confidence = probs[predicted_label]
                
                # Get top 3 predictions
                top_indices = np.argsort(probs)[-3:][::-1]
                top_predictions = [(self.emotion_labels[idx], probs[idx]) for idx in top_indices]
                
                result = {
                    'text': text,
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'top_predictions': top_predictions,
                    'all_probabilities': {self.emotion_labels[j]: probs[j] for j in range(len(probs))}
                }
                results.append(result)
                
                print(f"\n📝 Text: \"{text}\"")
                print(f"🎭 Predicted Emotion: {predicted_emotion} ({confidence:.3f})")
                print(f"🏆 Top 3 Predictions:")
                for emotion, prob in top_predictions:
                    print(f"   - {emotion}: {prob:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            raise
    
    def save_model(self, save_path="./saved_emotion_model"):
        """Save the trained model and tokenizer."""
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"💾 Model saved to: {save_path}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_saved_model(self, model_path="./saved_emotion_model"):
        """Load a previously saved model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            print(f"📥 Model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading saved model: {str(e)}")


def demo_usage():
    """Demonstrate how to use the trained model for emotion prediction."""
    print("\n" + "="*60)
    print("🎭 EMOTION PREDICTION DEMO")
    print("="*60)
    
    # Sample texts for prediction
    sample_texts = [
        "I am so happy today! The weather is beautiful and I got great news!",
        "I'm really scared about the upcoming exam. What if I fail?",
        "I love spending time with my family during the holidays.",
        "This traffic jam is making me so angry! I'm going to be late again.",
        "I feel so sad about leaving my hometown. I'll miss my friends.",
        "What a surprise! I never expected to see you here!"
    ]
    
    return sample_texts


def main():
    """Main function to run the complete emotion detection pipeline."""
    print("🎭 EMOTION DETECTION WITH TRANSFORMER MODELS")
    print("=" * 60)
    
    try:
        # Initialize the classifier
        classifier = EmotionClassifier(
            model_name="bert-base-uncased",  # You can change to "roberta-base"
            max_length=128
        )
        
        # Load model and tokenizer
        classifier.load_model_and_tokenizer()
        
        # Load and explore dataset
        dataset = classifier.load_dataset()
        
        # Preprocess the dataset
        tokenized_dataset = classifier.preprocess_dataset(dataset)
        
        # Setup training
        classifier.setup_training(tokenized_dataset)
        
        # Train the model
        train_result = classifier.train_model()
        
        # Evaluate the model
        eval_results, y_true, y_pred = classifier.evaluate_model(tokenized_dataset)
        
        # Save the trained model
        classifier.save_model()
        
        # Demo: Predict emotions for sample texts
        sample_texts = demo_usage()
        predictions = classifier.predict_emotion(sample_texts)
        
        print("\n🎉 Training and evaluation completed successfully!")
        print(f"🏆 Final Test Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"🎯 Final Test F1 Score: {eval_results['eval_f1']:.4f}")
        
        return classifier
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the complete pipeline
    trained_classifier = main()
    
    print("\n" + "="*60)
    print("✨ You can now use the trained model to predict emotions!")
    print("Example usage:")
    print("trained_classifier.predict_emotion('Your text here')")
    print("="*60)
