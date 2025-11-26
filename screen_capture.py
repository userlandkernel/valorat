import numpy as np
import mss
import cv2

class ScreenCapture:
    def __init__(self, config):
        self.config = config
        self.monitor = self.setup_monitor()
        self.sct = mss.mss()
    
    def setup_monitor(self):
        region = self.config['capture']['region']
        return {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3]
        }
    
    def capture(self):
        try:
            # Capture screen
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return frame
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
