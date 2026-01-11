import cv2
import numpy as np
import os

def analyze_density(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return f"Could not load {img_path}"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Simple thresholding similar to the backend
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = thresh.shape
    density = cv2.countNonZero(thresh) / (w * h)
    
    # Also check horizontal projection peak
    projection = np.sum(thresh, axis=1) / (255 * w)
    max_peak = np.max(projection)
    
    return {
        "file": os.path.basename(img_path),
        "density": density,
        "max_peak": max_peak
    }

base_path = "/home/t68/.gemini/antigravity/brain/e99fb68c-d72f-43b4-b6bf-916d91b3b8e7/"
images = [
    "uploaded_image_0_1768076828568.png",
    "uploaded_image_1_1768076828568.png",
    "uploaded_image_2_1768076828568.png"
]

for img_name in images:
    res = analyze_density(os.path.join(base_path, img_name))
    print(res)
