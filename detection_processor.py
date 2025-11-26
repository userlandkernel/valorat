import numpy as np
class DetectionProcessor:
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config['model']['confidence_threshold']
    
    def process_detections(self, detections, frame_shape):
        """Process model detections and extract enemy positions"""
        targets = []
        
        if hasattr(detections, 'xyxy'):
            # YOLO format
            for detection in detections.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                if conf > self.confidence_threshold and self.is_enemy(cls):
                    target = self.calculate_head_position(x1, y1, x2, y2, frame_shape)
                    targets.append({
                        'position': target,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return targets
    
    def is_enemy(self, class_id):
        """Determine if detection is an enemy player"""
        # Modify this based on your model's class mapping
        # For YOLO, person class is typically 0
        return class_id == 0
    
    def calculate_head_position(self, x1, y1, x2, y2, frame_shape):
        """Calculate head position from bounding box"""
        # Estimate head position (upper middle of bounding box)
        head_x = (x1 + x2) / 2
        head_y = y1 + (y2 - y1) * 0.2  # Adjust based on your model
        
        # Apply headshot offset
        head_y += self.config['aim']['headshot_offset']
        
        # Convert to screen coordinates
        screen_x = int(head_x * frame_shape[1] / self.config['model']['input_size'][0])
        screen_y = int(head_y * frame_shape[0] / self.config['model']['input_size'][1])
        
        return (screen_x, screen_y)
    
    def select_best_target(self, targets):
        """Select the best target based on confidence and position"""
        if not targets:
            return None
        
        # Sort by confidence (highest first)
        targets.sort(key=lambda x: x['confidence'], reverse=True)
        
        # For simplicity, return the highest confidence target
        # You could add distance-based selection here
        print(targets[0])
        return targets[0]
