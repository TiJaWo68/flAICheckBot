import cv2
import numpy as np
from PIL import Image
import os

def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
    if len(angles) > 0:
        median_angle = np.median(angles)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

def segment_lines_advanced(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Failed to load image"
    
    # 0. Deskew first
    img = deskew(img)
    cv2.imwrite("debug_8_deskewed.png", img)
    
    # Crop margin
    margin = 5
    h, w = img.shape[:2]
    img_cropped = img[margin:h-margin, margin:w-margin]
    
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    # 1. Remove ruled lines more aggressively
    # Detect horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    # Dilate slightly vertically to make sure we cover the whole line
    horiz_lines = cv2.dilate(horiz_lines, np.ones((3, 1), np.uint8), iterations=1)
    
    # Subtract lines from thresholded image
    thresh_no_lines = cv2.subtract(thresh, horiz_lines)
    cv2.imwrite("debug_9_no_lines.png", thresh_no_lines)
    
    # 2. Dilation with very small vertical kernel
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    dilated = cv2.dilate(thresh_no_lines, line_kernel, iterations=2)
    cv2.imwrite("debug_10_dilated_final.png", dilated)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        x, y, wc, hc = cv2.boundingRect(cnt)
        # Filter noise
        if wc < 30 or hc < 8:
            continue
        valid_contours.append(cnt)
        
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])
    
    result_img = img_cropped.copy()
    for i, cnt in enumerate(valid_contours):
        x, y, wc, hc = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x+wc, y+hc), (0, 255, 0), 2)
        cv2.putText(result_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imwrite("debug_11_results_final.png", result_img)
    return f"Detected {len(valid_contours)} lines."

image_path = "/home/t68/eclipse-workspace/flAICheckBot/exported_samples/en/sample_20.png"
print(segment_lines_advanced(image_path))
