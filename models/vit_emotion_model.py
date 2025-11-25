"""
Vision Transformer model for facial emotion recognition.
Uses pretrained ViT from Hugging Face as backbone with custom classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from typing import Optional, Dict, Any


class ViTEmotionClassifier(nn.Module):
    """
    Vision Transformer for Facial Emotion Classification.
    
    This model uses a pretrained ViT backbone from Hugging Face and adds
    a custom classification head for emotion prediction.
    """
    
    def __init__(
        self, 
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 7,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize the ViT emotion classifier.
        
        Args:
            model_name: Name of the pretrained ViT model from Hugging Face
            num_classes: Number of emotion classes (default: 7 for basic emotions)
            dropout_rate: Dropout rate for the classifier head
            freeze_backbone: Whether to freeze the ViT backbone during training
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Get hidden size from ViT config
        hidden_size = self.vit.config.hidden_size
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier head weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, pixel_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input image tensor [batch_size, 3, 224, 224]
            attention_mask: Optional attention mask
            
        Returns:
            logits: Emotion class logits [batch_size, num_classes]
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values, attention_mask=attention_mask)
        
        # Extract CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Pass through classifier
        logits = self.classifier(cls_embedding)
        
        return logits
    
    def predict_emotion(self, pixel_values: torch.Tensor, emotion_labels: Optional[list] = None):
        """
        Predict emotion with human-readable labels.
        
        Args:
            pixel_values: Input image tensor
            emotion_labels: List of emotion label names
            
        Returns:
            Dictionary with predicted class, confidence, and probabilities
        """
        if emotion_labels is None:
            emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            # Convert to numpy for easier handling
            if predicted_class.dim() == 0:  # Single prediction
                pred_idx = predicted_class.item()
                conf = confidence.item()
                probs = probabilities.squeeze().cpu().numpy()
                
                return {
                    'predicted_emotion': emotion_labels[pred_idx],
                    'confidence': conf,
                    'probabilities': {label: prob for label, prob in zip(emotion_labels, probs)}
                }
            else:  # Batch predictions
                predictions = []
                for i in range(len(predicted_class)):
                    pred_idx = predicted_class[i].item()
                    conf = confidence[i].item()
                    probs = probabilities[i].cpu().numpy()
                    
                    predictions.append({
                        'predicted_emotion': emotion_labels[pred_idx],
                        'confidence': conf,
                        'probabilities': {label: prob for label, prob in zip(emotion_labels, probs)}
                    })
                return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_frozen': trainable_params < total_params * 0.5
        }


class ViTEmotionEnsemble(nn.Module):
    """
    Ensemble of multiple ViT models for improved accuracy.
    """
    
    def __init__(self, model_configs: list, weights: Optional[list] = None):
        """
        Initialize ensemble of ViT emotion models.
        
        Args:
            model_configs: List of model configuration dictionaries
            weights: Optional weights for ensemble averaging
        """
        super().__init__()
        
        self.models = nn.ModuleList([
            ViTEmotionClassifier(**config) for config in model_configs
        ])
        
        self.weights = weights if weights else [1.0] * len(self.models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)  # Normalize
    
    def forward(self, pixel_values: torch.Tensor):
        """Forward pass through ensemble."""
        outputs = []
        for model in self.models:
            logits = model(pixel_values)
            outputs.append(logits)
        
        # Weighted average
        weighted_output = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_output
