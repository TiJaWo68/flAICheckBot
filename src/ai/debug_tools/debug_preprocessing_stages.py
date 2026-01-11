import cv2
import numpy as np
import os
from PIL import Image

def deskew(img: np.ndarray) -> np.ndarray:
    """Corrects the skew of the image."""
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

def split_lines_by_projection_debug(roi_thresh: np.ndarray, roi_color: np.ndarray, y_offset, x_offset):
    """Debug version of splitting that returns bounding boxes."""
    projection = np.sum(roi_thresh, axis=1)
    width = roi_thresh.shape[1]
    height = roi_thresh.shape[0]
    norm_projection = projection / (255 * width)
    
    gap_threshold = 0.01 
    lines_bboxes = []
    in_line = False
    start_y = 0
    
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(norm_projection, kernel, mode='same')
    
    for y, val in enumerate(smoothed):
        if not in_line and val > gap_threshold:
            in_line = True
            start_y = y
        elif in_line and val <= gap_threshold:
            in_line = False
            if y - start_y > 8:
                y_low = max(0, start_y - 2)
                y_high = min(height, y + 2)
                
                sub_thresh = roi_thresh[y_low:y_high, :]
                sub_density = cv2.countNonZero(sub_thresh) / (width * (y_high - y_low))
                if sub_density > 0.015: 
                    lines_bboxes.append((x_offset, y_offset + y_low, width, y_high - y_low))
    
    if in_line and height - start_y > 8:
        lines_bboxes.append((x_offset, y_offset + start_y, width, height - start_y))
        
    return lines_bboxes

def detect_ruled_lines(thresh: np.ndarray) -> list[int]:
    """Detects horizontal writing lines and returns their Y-coordinates."""
    h, w = thresh.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    proj = np.sum(detected, axis=1)
    y_coords = []
    threshold = np.max(proj) * 0.5
    for y, val in enumerate(proj):
        if val > threshold:
            window = proj[max(0, y-3):min(h, y+4)]
            if val == np.max(window):
                y_coords.append(y)
    if not y_coords: return []
    filtered_y = [y_coords[0]]
    for i in range(1, len(y_coords)):
        if y_coords[i] - filtered_y[-1] > 10:
            filtered_y.append(y_coords[i])
    return filtered_y

def apply_schwarzmaske(crop: np.ndarray, padding_bottom: int) -> np.ndarray:
    """Removes connected components that are entirely in the bottom padding area."""
    h, w = crop.shape[:2]
    if h < 10 or w < 10 or padding_bottom <= 0: return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    safe_y_limit = h - padding_bottom
    mask = np.ones_like(thresh) * 255
    for i in range(1, num_labels):
        comp_y = stats[i, cv2.CC_STAT_TOP]
        if comp_y < safe_y_limit: mask[labels == i] = 0
    result = crop.copy()
    result[mask == 255] = [255, 255, 255]
    return result

def remove_ruling_lines(crop: np.ndarray) -> np.ndarray:
    """Removes horizontal and vertical ruling line artifacts from the crop."""
    h, w = crop.shape[:2]
    if h < 5 or w < 5: return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horiz_mask = horiz_lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vert_mask = vert_lines
    mask = cv2.bitwise_or(horiz_mask, vert_mask)
    result = crop.copy()
    result[mask > 0] = [255, 255, 255]
    return result

def debug_preprocess(img_path, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    img = cv2.imread(img_path)
    if img is None: return

    cv2.imwrite(os.path.join(output_dir, "step0_original.png"), img)
    deskewed = deskew(img)
    cv2.imwrite(os.path.join(output_dir, "step1_deskewed.png"), deskewed)

    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    ruled_y = detect_ruled_lines(thresh)
    viz_guides = deskewed.copy()
    for y in ruled_y:
        cv2.line(viz_guides, (0, y), (viz_guides.shape[1], y), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "step3_guides.png"), viz_guides)

    h_img, w_img = deskewed.shape[:2]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 15, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 15))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    horiz_mask = cv2.dilate(horiz_lines, np.ones((5, 1), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert_lines, np.ones((1, 5), np.uint8), iterations=1)
    thresh_clean = cv2.subtract(thresh, horiz_mask)
    thresh_clean = cv2.subtract(thresh_clean, vert_mask)
    
    cv2.imwrite(os.path.join(output_dir, "step5_thresh_no_lines.png"), cv2.bitwise_not(thresh_clean))

    final_viz = deskewed.copy()
    lines_dir = os.path.join(output_dir, "lines")
    if os.path.exists(lines_dir): import shutil; shutil.rmtree(lines_dir)
    os.makedirs(lines_dir)
    
    line_count = 0
    if len(ruled_y) >= 2:
        distances = [ruled_y[i] - ruled_y[i-1] for i in range(1, len(ruled_y))]
        median_dist = int(np.median(distances))
        padding = int(median_dist * 0.25)
        
        if ruled_y[0] > 10: ruled_y.insert(0, 5)
        if h_img - ruled_y[-1] > median_dist: ruled_y.append(ruled_y[-1] + median_dist)
            
        for i in range(len(ruled_y) - 1):
            y_start = ruled_y[i]
            y_end = ruled_y[i+1]
            y_low = y_start
            y_high = min(h_img, y_end + padding)
            
            roi_thresh = thresh_clean[y_low:y_high, :]
            density = cv2.countNonZero(roi_thresh) / (w_img * (y_high - y_low))
            
            if density > 0.01:
                line_count += 1
                cv2.rectangle(final_viz, (0, y_low), (w_img, y_high), (0, 255, 0), 2)
                line_crop = deskewed[y_low:y_high, :]
                # Iteration 6: Remove lines then mask
                clean_crop = remove_ruling_lines(line_crop)
                masked_crop = apply_schwarzmaske(clean_crop, padding_bottom=padding)
                cv2.imwrite(os.path.join(lines_dir, f"line_{line_count:02d}.png"), masked_crop)
    
    cv2.imwrite(os.path.join(output_dir, "step7_segmentation.png"), final_viz)
    print(f"Debug images saved to {output_dir}. Total lines: {line_count}")
    
    cv2.imwrite(os.path.join(output_dir, "step7_segmentation.png"), final_viz)
    print(f"Debug images saved to {output_dir}. Total lines: {line_count}")
    
    cv2.imwrite(os.path.join(output_dir, "step7_segmentation.png"), final_viz)
    print(f"Debug images saved to {output_dir}. Total lines: {line_count}")

if __name__ == "__main__":
    path = "/home/t68/eclipse-workspace/flAICheckBot/exported_samples/en/sample_19.png"
    out = "/home/t68/eclipse-workspace/flAICheckBot/debug_preproc_sample19"
    debug_preprocess(path, out)
