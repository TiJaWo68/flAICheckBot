
import os
import sys
import cv2
import numpy as np

sys.path.append("/home/t68/eclipse-workspace/flAICheckBot/src/ai")
import icr_prototype

def check_dims():
    img_path = "/home/t68/eclipse-workspace/flAICheckBot/exported_samples/en/sample_20.png"
    img = cv2.imread(img_path)
    
    lines = icr_prototype.segment_lines(img)
    print(f"Detected {len(lines)} lines.\n")
    
    print("--- Dimensions of first 5 lines ---")
    for i, line_img in enumerate(lines[:5]):
        w, h = line_img.size
        print(f"Line {i}: {w}x{h} (Aspect Ratio: {w/h:.2f})")
        
        # Simulate 'pad_image' logic
        target_size = (384, 384)
        ratio = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        print(f"  -> Resized to fit 384x384: {new_w}x{new_h}")
        if new_h < 30:
            print("  [WARNING] Text height is likely too small for recognition!")

if __name__ == "__main__":
    check_dims()
