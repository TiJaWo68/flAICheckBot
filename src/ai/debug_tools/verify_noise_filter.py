import cv2
import numpy as np
import os
import sys

# Add current directory to path so we can import icr_prototype
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import icr_prototype

def test_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return f"Could not load {img_path}"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Backend thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Analyze using the actual production function
    result = icr_prototype.is_noise_segment(img, thresh)
    
    # Gather stats for debugging
    avg_brightness = np.mean(img)
    projection_v = np.sum(thresh, axis=1) / (255 * thresh.shape[1])
    max_peak = np.max(projection_v)
    std_dev = np.std(img)
    
    return {
        "file": os.path.basename(img_path),
        "is_noise": result,
        "avg_brightness": avg_brightness,
        "max_peak": max_peak,
        "std_dev": std_dev
    }

base_path = "/home/t68/.gemini/antigravity/brain/e99fb68c-d72f-43b4-b6bf-916d91b3b8e7/"
images = [
    "uploaded_image_0_1768076828568.png",
    "uploaded_image_1_1768076828568.png",
    "uploaded_image_2_1768076828568.png"
]

print("--- Start Verification ---")
for img_name in images:
    res = test_image(os.path.join(base_path, img_name))
    print(res)
print("--- End Verification ---")
