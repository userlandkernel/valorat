import time
import cv2
import numpy as np
import torch
import keyboard
from model_loader import ModelLoader
from screen_capture import ScreenCapture
from aim_controller import AimController
from detection_processor import DetectionProcessor
import yaml

class ValorantAimbot:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.running = False
        self.model_loader = ModelLoader(self.config)
        self.screen_capture = ScreenCapture(self.config)
        self.aim_controller = AimController(self.config)
        self.detection_processor = DetectionProcessor(self.config)
        
        self.setup_hotkeys()
        
    def setup_hotkeys(self):
        keyboard.on_press_key(self.config['safety']['toggle_key'], 
                            lambda _: self.toggle())
        keyboard.on_press_key(self.config['safety']['exit_key'], 
                            lambda _: self.stop())
    
    def toggle(self):
        self.running = not self.running
        print(f"Aimbot {'enabled' if self.running else 'disabled'}")
    
    def stop(self):
        self.running = False
        print("Shutting down...")
        exit()
    
    def run(self):
        print("Valorant Aimbot started. Press F6 to toggle, F7 to exit.")
        
        while True:
            if self.running:
                self.process_frame()
            
            time.sleep(1/self.config['capture']['fps'])
    
    def process_frame(self):
        # Capture screen
        frame = self.screen_capture.capture()
        if frame is None:
            return
        
        # Detect enemies
        detections = self.model_loader.detect(frame)
        
        # Process detections
        targets = self.detection_processor.process_detections(detections, frame.shape)
        
        # Aim at best target
        if targets:
            best_target = self.detection_processor.select_best_target(targets)
            self.aim_controller.aim_at_target(best_target)

if __name__ == "__main__":
    aimbot = ValorantAimbot()
    aimbot.run()
