"""
Inference script for trained ViT emotion recognition model.
Supports single image prediction, batch processing, and real-time webcam inference.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.transforms import get_inference_transforms, TenCropTransform


class EmotionPredictor:
    """Predictor class for emotion inference."""
    
    def __init__(self, 
                 model: nn.Module,
                 transform,
                 device: torch.device,
                 emotion_labels: List[str]):
        """
        Initialize predictor.
        
        Args:
            model: Trained PyTorch model
            transform: Image transformation pipeline
            device: Device to run inference on
            emotion_labels: List of emotion label names
        """
        self.model = model
        self.transform = transform
        self.device = device
        self.emotion_labels = emotion_labels
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict_single_image(self, 
                           image_path: str, 
                           return_probabilities: bool = True,
                           use_tta: bool = False) -> Dict:
        """
        Predict emotion for a single image.
        
        Args:
            image_path: Path to image file
            return_probabilities: Whether to return class probabilities
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        if use_tta:
            # Test-time augmentation with TenCrop
            tta_transform = TenCropTransform()
            image_crops = tta_transform(image)  # Returns tensor of shape [10, 3, 224, 224]
            image_crops = image_crops.to(self.device)
            
            with torch.no_grad():
                # Get predictions for all crops
                outputs = self.model(image_crops)
                # Average predictions
                avg_output = outputs.mean(dim=0, keepdim=True)
                probabilities = torch.softmax(avg_output, dim=1)
                predicted_class = torch.argmax(avg_output, dim=1)
        else:
            # Standard single crop prediction
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1)
        
        # Extract results
        pred_idx = predicted_class.item()
        confidence = probabilities[0, pred_idx].item()
        predicted_emotion = self.emotion_labels[pred_idx]
        
        result = {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'predicted_class_id': pred_idx
        }
        
        if return_probabilities:
            probs_dict = {
                emotion: float(probabilities[0, i])
                for i, emotion in enumerate(self.emotion_labels)
            }
            result['probabilities'] = probs_dict
        
        return result
    
    def predict_batch(self, 
                     image_paths: List[str],
                     batch_size: int = 32,
                     return_probabilities: bool = True) -> List[Dict]:
        """
        Predict emotions for a batch of images.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load and preprocess batch
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Add placeholder result for failed images
                    results.append({
                        'predicted_emotion': 'unknown',
                        'confidence': 0.0,
                        'predicted_class_id': -1,
                        'error': str(e)
                    })
            
            if batch_images:
                # Stack images into batch tensor
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(outputs, dim=1)
                
                # Process results
                for k, valid_idx in enumerate(valid_indices):
                    pred_idx = predicted_classes[k].item()
                    confidence = probabilities[k, pred_idx].item()
                    predicted_emotion = self.emotion_labels[pred_idx]
                    
                    result = {
                        'predicted_emotion': predicted_emotion,
                        'confidence': confidence,
                        'predicted_class_id': pred_idx,
                        'image_path': batch_paths[valid_idx]
                    }
                    
                    if return_probabilities:
                        probs_dict = {
                            emotion: float(probabilities[k, i])
                            for i, emotion in enumerate(self.emotion_labels)
                        }
                        result['probabilities'] = probs_dict
                    
                    # Insert result at correct position
                    result_idx = i + valid_idx
                    if result_idx < len(results):
                        results[result_idx] = result
                    else:
                        results.append(result)
        
        return results
    
    def predict_webcam(self, 
                      face_cascade_path: str = None,
                      display_probabilities: bool = True,
                      save_video: str = None):
        """
        Real-time emotion prediction from webcam.
        
        Args:
            face_cascade_path: Path to OpenCV face cascade file
            display_probabilities: Whether to display probability bars
            save_video: Path to save output video (optional)
        """
        # Initialize face detector
        if face_cascade_path is None:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Video writer for saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (640, 480))
        
        print("Starting webcam emotion detection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    try:
                        # Convert to PIL Image and predict
                        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        prediction = self.predict_single_image_from_pil(face_pil)
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display prediction
                        emotion = prediction['predicted_emotion']
                        confidence = prediction['confidence']
                        label = f"{emotion}: {confidence:.2f}"
                        
                        cv2.putText(frame, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Display probability bars if requested
                        if display_probabilities and 'probabilities' in prediction:
                            self.draw_probability_bars(frame, prediction['probabilities'], x, y+h+10)
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                
                # Save frame if video writer is active
                if video_writer:
                    video_writer.write(frame)
                
                # Display frame
                cv2.imshow('Emotion Detection', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
    
    def predict_single_image_from_pil(self, pil_image: Image.Image) -> Dict:
        """Predict emotion from PIL Image object."""
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
        
        pred_idx = predicted_class.item()
        confidence = probabilities[0, pred_idx].item()
        predicted_emotion = self.emotion_labels[pred_idx]
        
        result = {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'predicted_class_id': pred_idx,
            'probabilities': {
                emotion: float(probabilities[0, i])
                for i, emotion in enumerate(self.emotion_labels)
            }
        }
        
        return result
    
    def draw_probability_bars(self, frame, probabilities: Dict, x: int, y: int):
        """Draw probability bars on frame."""
        bar_width = 200
        bar_height = 15
        
        for i, (emotion, prob) in enumerate(probabilities.items()):
            bar_y = y + i * (bar_height + 5)
            
            # Draw background bar
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Draw probability bar
            prob_width = int(bar_width * prob)
            color = (0, int(255 * prob), int(255 * (1 - prob)))
            cv2.rectangle(frame, (x, bar_y), (x + prob_width, bar_y + bar_height), 
                         color, -1)
            
            # Draw text
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (x + bar_width + 10, bar_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model info from checkpoint
    emotion_labels = checkpoint.get('emotion_labels', 
                                  ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    num_classes = len(emotion_labels)
    
    # Create model
    model = ViTEmotionClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Emotion labels: {emotion_labels}")
    
    return model, emotion_labels


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='ViT Emotion Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'webcam'], default='single',
                       help='Inference mode')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch mode')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--save_video', type=str, help='Path to save webcam video')
    parser.add_argument('--face_cascade', type=str, help='Path to face cascade file')
    
    args = parser.parse_args()
    
    # Device - prioritize MPS for Apple Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: CUDA GPU')
    else:
        device = torch.device('cpu')
        print(f'Using device: CPU')
    
    print(f'Device: {device}')
    
    # Load model
    model, emotion_labels = load_model(args.checkpoint, device)
    
    # Data transforms
    transform = get_inference_transforms()
    
    # Create predictor
    predictor = EmotionPredictor(
        model=model,
        transform=transform,
        device=device,
        emotion_labels=emotion_labels
    )
    
    # Run inference based on mode
    if args.mode == 'single':
        if not args.input:
            raise ValueError("Input image path required for single mode")
        
        print(f"Predicting emotion for: {args.input}")
        result = predictor.predict_single_image(args.input, use_tta=args.use_tta)
        
        print(f"\nPrediction Results:")
        print(f"Predicted emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if 'probabilities' in result:
            print(f"\nAll probabilities:")
            for emotion, prob in result['probabilities'].items():
                print(f"  {emotion:>10}: {prob:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.mode == 'batch':
        if not args.input:
            raise ValueError("Input directory required for batch mode")
        
        # Get all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file in os.listdir(args.input):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.input, file))
        
        print(f"Found {len(image_paths)} images in {args.input}")
        
        # Run predictions
        start_time = time.time()
        results = predictor.predict_batch(image_paths, batch_size=args.batch_size)
        elapsed_time = time.time() - start_time
        
        print(f"Processed {len(results)} images in {elapsed_time:.2f}s")
        print(f"Average time per image: {elapsed_time/len(results):.3f}s")
        
        # Print summary
        emotion_counts = {}
        for result in results:
            if 'predicted_emotion' in result:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion:>10}: {count} ({count/len(results)*100:.1f}%)")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.mode == 'webcam':
        predictor.predict_webcam(
            face_cascade_path=args.face_cascade,
            save_video=args.save_video
        )


if __name__ == '__main__':
    main()
