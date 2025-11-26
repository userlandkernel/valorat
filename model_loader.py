import torch
import torchvision
import numpy as np
import cv2

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load your custom trained model"""
        try:
            # For PyTorch models
            model = torch.load(self.config['model']['path'])
            model.eval()
            print("Custom model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to YOLO if custom model fails
            return self.load_fallback_model()
    
    def load_fallback_model(self):
        """Fallback to pre-trained YOLOv5 for object detection"""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def detect(self, frame):
        """Perform object detection on frame"""
        # Preprocess frame
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                # Custom model inference
                results = self.model.predict(input_tensor)
            else:
                # YOLO inference
                results = self.model(input_tensor)
        
        return results
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to model input size
        input_size = self.config['model']['input_size']
        frame_resized = cv2.resize(frame, tuple(input_size))
        
        # Normalize and convert to tensor
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(frame_normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
        
        return tensor
