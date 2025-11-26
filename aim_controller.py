import pyautogui
import time
import numpy as np

class AimController:
    def __init__(self, config):
        self.config = config
        self.smoothing = config['aim']['smoothing']
        self.reaction_delay = config['aim']['reaction_delay']
        
        # Safety measures
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
    
    def aim_at_target(self, target):
        """Move mouse to target position with smoothing"""
        if not target:
            return
        
        target_x, target_y = target['position']
        
        # Get current mouse position
        current_x, current_y = pyautogui.position()
        
        # Calculate movement with smoothing
        move_x = current_x + (target_x - current_x) * self.smoothing
        move_y = current_y + (target_y - current_y) * self.smoothing
        
        # Add reaction delay
        time.sleep(self.reaction_delay)
        
        # Move mouse
        pyautogui.moveTo(move_x, move_y, duration=0.01)
        
        # Auto-shoot (optional - use with caution)
        # pyautogui.click()
