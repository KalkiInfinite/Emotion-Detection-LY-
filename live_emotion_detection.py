"""
Real-time Emotion Recognition using Laptop Camera
Uses your trained ViT model to classify emotions from live camera feed.
"""

import cv2
import torch
import numpy as np
import time
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vit_emotion_model import ViTEmotionClassifier
from data.transforms import get_val_transforms


class LiveEmotionDetector:
    """Real-time emotion detection using camera feed."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the live emotion detector.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'mps', 'cuda', 'cpu')
        """
        # Emotion labels (based on your dataset) - must be set before loading model
        self.emotion_labels = [
            'angry', 'contempt', 'disgust', 'fear', 
            'happy', 'neutral', 'sad', 'surprise'
        ]
        
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = get_val_transforms()
        
        # Colors for each emotion (BGR format for OpenCV)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'contempt': (0, 100, 255), # Orange
            'disgust': (0, 255, 0),    # Green
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 255, 0), # Cyan
        }
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"🎥 Live Emotion Detector initialized")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Emotions: {', '.join(self.emotion_labels)}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> ViTEmotionClassifier:
        """Load the trained model."""
        print(f"📂 Loading model from: {model_path}")
        
        # Create model instance
        model = ViTEmotionClassifier(
            model_name='google/vit-base-patch16-224',
            num_classes=len(self.emotion_labels)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
    
    def preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for ViT model.
        
        Args:
            face_img: Face image as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict_emotion(self, face_tensor: torch.Tensor) -> tuple:
        """
        Predict emotion from face tensor.
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_emotion = self.emotion_labels[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probs = probabilities.squeeze().cpu().numpy()
            
            return predicted_emotion, confidence_score, all_probs
    
    def draw_results(self, frame: np.ndarray, x: int, y: int, w: int, h: int, 
                    emotion: str, confidence: float, all_probs: np.ndarray) -> np.ndarray:
        """
        Draw emotion prediction results on frame.
        
        Args:
            frame: Original frame
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion
            confidence: Confidence score
            all_probs: All emotion probabilities
            
        Returns:
            Frame with drawn results
        """
        # Get emotion color
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label and confidence
        label = f"{emotion}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw probability bars for top 3 emotions
        top_3_indices = np.argsort(all_probs)[-3:][::-1]
        bar_width = 200
        bar_height = 20
        start_x = x + w + 10
        start_y = y
        
        for i, idx in enumerate(top_3_indices):
            emotion_name = self.emotion_labels[idx]
            prob = all_probs[idx]
            
            # Bar background
            cv2.rectangle(frame, (start_x, start_y + i * 25), 
                         (start_x + bar_width, start_y + i * 25 + bar_height), 
                         (50, 50, 50), -1)
            
            # Bar fill
            fill_width = int(bar_width * prob)
            bar_color = self.emotion_colors.get(emotion_name, (255, 255, 255))
            cv2.rectangle(frame, (start_x, start_y + i * 25), 
                         (start_x + fill_width, start_y + i * 25 + bar_height), 
                         bar_color, -1)
            
            # Emotion label and percentage
            text = f"{emotion_name}: {prob:.1%}"
            cv2.putText(frame, text, (start_x + 5, start_y + i * 25 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id: int = 0, window_name: str = "Live Emotion Detection"):
        """
        Run live emotion detection.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            window_name: Name of the display window
        """
        print(f"🚀 Starting live emotion detection...")
        print(f"📷 Camera ID: {camera_id}")
        print(f"⌨️  Press 'q' to quit, 's' to save screenshot")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        screenshot_counter = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    try:
                        # Extract face region
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Preprocess face
                        face_tensor = self.preprocess_face(face_img)
                        
                        # Predict emotion
                        emotion, confidence, all_probs = self.predict_emotion(face_tensor)
                        
                        # Draw results
                        frame = self.draw_results(frame, x, y, w, h, 
                                                emotion, confidence, all_probs)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing face: {e}")
                        # Draw simple rectangle for failed face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Processing Error", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Draw FPS and instructions
                cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_name = f"emotion_screenshot_{screenshot_counter:03d}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 Screenshot saved: {screenshot_name}")
                    screenshot_counter += 1
        
        except KeyboardInterrupt:
            print("\n⌨️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("🎬 Camera released and windows closed")


def main():
    """Main function to run live emotion detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Emotion Detection using Camera')
    parser.add_argument('--model_path', type=str, 
                       default='./outputs/safe_training/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Device for inference (default: auto)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Available model files:")
        model_dir = os.path.dirname(args.model_path)
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pt'):
                    print(f"   - {os.path.join(model_dir, file)}")
        return
    
    # Create detector
    detector = LiveEmotionDetector(args.model_path, args.device)
    
    # Run detection
    detector.run(args.camera_id)


if __name__ == "__main__":
    main()
