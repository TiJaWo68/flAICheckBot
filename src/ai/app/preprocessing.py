import cv2
import numpy as np
from PIL import Image

def deskew(img: np.ndarray) -> np.ndarray:
    """Corrects the skew of the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary thresholding for edge detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use Hough Transform to find lines
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45: # Filter out vertical/extreme lines
                angles.append(angle)
    
    if len(angles) > 0:
        median_angle = np.median(angles)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return img

def detect_ruled_lines(thresh: np.ndarray) -> list[int]:
    """Detects horizontal writing lines and returns their Y-coordinates."""
    h, w = thresh.shape[:2]
    # Use a long horizontal kernel to find ruled lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    detected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Sum across width to find peaks
    proj = np.sum(detected, axis=1)
    # Filter peaks to get precise line positions
    y_coords = []
    threshold = np.max(proj) * 0.5
    for y, val in enumerate(proj):
        if val > threshold:
            # Simple peak picking: if it's the max in a small window
            window = proj[max(0, y-3):min(h, y+4)]
            if val == np.max(window):
                y_coords.append(y)
    
    # Filter out lines that are too close to each other (duplicates)
    if not y_coords:
        return []
        
    filtered_y = [y_coords[0]]
    for i in range(1, len(y_coords)):
        if y_coords[i] - filtered_y[-1] > 10:
            filtered_y.append(y_coords[i])
            
    return filtered_y

def advanced_preprocess(img: np.ndarray) -> np.ndarray:
    """Denoising, adaptive binarization, and deskewing."""
    # 1. Denoising - use smaller kernel to avoid blurring lines together
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. Deskewing
    deskewed = deskew(denoised)
    
    return deskewed

def apply_schwarzmaske(crop: np.ndarray, padding_bottom: int) -> np.ndarray:
    """Removes connected components that are entirely in the bottom padding area."""
    h, w = crop.shape[:2]
    if h < 10 or w < 10 or padding_bottom <= 0:
        return crop
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Safe zone is everything ABOVE the bottom padding
    safe_y_limit = h - padding_bottom
    
    mask = np.ones_like(thresh) * 255 # White mask (pixels to remove)
    
    for i in range(1, num_labels):
        comp_y = stats[i, cv2.CC_STAT_TOP]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Keep if any part of the component is in the safe zone (above the padding)
        # 0% padding at top means everything at top is safe.
        if comp_y < safe_y_limit:
            mask[labels == i] = 0 # Keep (don't mask)
            
    result = crop.copy()
    result[mask == 255] = [255, 255, 255]
    return result

def remove_ruling_lines(crop: np.ndarray) -> np.ndarray:
    """Removes horizontal and vertical ruling line artifacts from the crop."""
    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return crop
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Remove Horizontal Lines (e.g. at y=0 or y=h)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    # Iteration 7: Remove dilation to preserve letters
    horiz_mask = horiz_lines 
    
    # 2. Remove Vertical Lines (artifacts)
    # Use a tall kernel to find long vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    # Iteration 7: Remove dilation to preserve letters
    vert_mask = vert_lines
    
    mask = cv2.bitwise_or(horiz_mask, vert_mask)
    
    result = crop.copy()
    result[mask > 0] = [255, 255, 255]
    return result

def robust_horizontal_crop(roi_thresh: np.ndarray, breathing_room: int = 15) -> tuple[int, int]:
    """
    Finds the horizontal bounds of the main text block using projection,
    ignoring isolated noise in the margins.
    """
    h, w = roi_thresh.shape
    if w == 0:
        return 0, 0
    
    # Calculate horizontal projection (average ink per column)
    projection = np.sum(roi_thresh, axis=0) / 255
    
    # Aggressive threshold: demand at least 1.5 pixels average height
    # to avoid marginal noise or ruling line remnants.
    threshold = 1.5 
    
    valid_indices = np.where(projection > threshold)[0]
    
    if valid_indices.size == 0:
        # Fallback to a less aggressive threshold if nothing found
        valid_indices = np.where(projection > 0.5)[0]
        if valid_indices.size == 0:
            return 0, w
        
    start_x = max(0, int(valid_indices[0]) - breathing_room)
    end_x = min(w, int(valid_indices[-1]) + breathing_room)
    
    return start_x, end_x

def is_noise_segment(roi_color: np.ndarray, roi_thresh: np.ndarray) -> tuple[bool, str]:
    """
    Advanced noise rejection based on user request and diagnostic analysis.
    Returns True if the segment is likely noise/ruling-line/border.
    """
    if roi_color.size == 0:
        return True, "Empty"
    
    # 1. Average Brightness Check (User request)
    # Reject if it's nearly pure white (original image)
    avg_brightness = np.mean(roi_color)
    if avg_brightness > 248:
        return True, f"Brightness ({avg_brightness:.1f})"
        
    # 2. Peak Row Density Check (Ruling line/Border filter)
    # Real text is sparse. If a single row has > 80% ink density, it's an artifact.
    h, w = roi_thresh.shape
    if w > 0:
        projection_v = np.sum(roi_thresh, axis=1) / (255 * w)
        max_p = np.max(projection_v)
        if max_p > 0.8:
            return True, f"Peak Density ({max_p:.2f})"
            
    # 3. Standard Deviation Check (Variance filter)
    # Empty regions have very low variance
    v = np.std(roi_color)
    if v < 2.0:
        return True, f"Variance ({v:.2f} < 2.0)"
    
    return False, ""

def split_lines_by_projection(roi_thresh: np.ndarray, roi_color: np.ndarray) -> list[tuple[Image.Image, list[int], dict]]:
    """Uses horizontal projection to split a block into individual lines precisely."""
    projection = np.sum(roi_thresh, axis=1)
    
    width = roi_thresh.shape[1]
    height = roi_thresh.shape[0]
    # Normalize by max possible ink in a row
    norm_projection = projection / (255 * width)
    
    # Sophisticated peak detection: find regions where density > threshold
    # and gaps that are significantly large
    gap_threshold = 0.01 # Very sensitive to small ink traces
    
    lines = []
    in_line = False
    start_y = 0
    
    # Smoothing for the 1D projection to bridge small vertical gaps in handwriting
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(norm_projection, kernel, mode='same')
    
    for y, val in enumerate(smoothed):
        if not in_line and val > gap_threshold:
            in_line = True
            start_y = y
        elif in_line and val <= gap_threshold:
            in_line = False
            # Filter out very thin fragments (noise/ruling remnants)
            if y - start_y > 8:
                # Add margin to the ROI
                y_low = max(0, start_y - 2)
                y_high = min(height, y + 2)
                line_roi = roi_color[y_low:y_high, :]
                
                # Internal Density Check and Tight Cropping
                sub_thresh = roi_thresh[y_low:y_high, :]
                
                # Find bounds of ink in this segment using horizontal projection
                x_s, x_e = robust_horizontal_crop(sub_thresh, breathing_room=10)
                
                # Vertical check for ink
                coords_y = np.where(np.any(sub_thresh[:, x_s:x_e] > 0, axis=1))[0]
                if coords_y.size > 0:
                    y_s_rel, y_e_rel = coords_y[0], coords_y[-1]
                    y_s = max(0, y_low + y_s_rel - 2)
                    y_e = min(height, y_low + y_e_rel + 2)
                else:
                    y_s, y_e = y_low, y_high

                sub_density = cv2.countNonZero(sub_thresh) / (width * (y_high - y_low))
                
                # Metadata for this segment
                meta = {"density": sub_density, "rejected": False, "reason": ""}

                if sub_density < 0.015:
                    meta["rejected"] = True
                    meta["reason"] = f"Density ({sub_density:.4f} < 0.015)"
                else:
                    # ADVANCED: User-requested noise rejection (Brightness, Peak Density, Variance)
                    is_noise, reason = is_noise_segment(line_roi, sub_thresh)
                    if is_noise:
                        meta["rejected"] = True
                        meta["reason"] = reason
                        
                line_roi_cropped = roi_color[y_s:y_e, x_s:x_e]
                img_line = Image.fromarray(cv2.cvtColor(line_roi_cropped, cv2.COLOR_BGR2RGB))
                # bbox: [x, y, width, height]
                lines.append((img_line, [int(x_s), int(y_s), int(x_e - x_s), int(y_e - y_s)], meta))
    
    # Handle trailing segment
    if in_line and height - start_y > 8:
        y_low = max(0, start_y - 2)
        y_high = height
        line_roi = roi_color[y_low:y_high, :]
        sub_thresh = roi_thresh[y_low:y_high, :]
        x_s, x_e = robust_horizontal_crop(sub_thresh, breathing_room=10)
        
        sub_density = cv2.countNonZero(sub_thresh) / (width * (y_high - y_low))
        meta = {"density": sub_density, "rejected": False, "reason": ""}
        if sub_density < 0.015:
            meta["rejected"] = True
            meta["reason"] = f"Density ({sub_density:.4f})"
        else:
            is_noise, reason = is_noise_segment(line_roi, sub_thresh)
            if is_noise:
                meta["rejected"] = True
                meta["reason"] = reason

        line_roi_cropped = roi_color[y_low:y_high, x_s:x_e]
        img_line = Image.fromarray(cv2.cvtColor(line_roi_cropped, cv2.COLOR_BGR2RGB))
        lines.append((img_line, [int(x_s), int(y_low), int(x_e - x_s), int(y_high - y_low)], meta))
    
    return lines

def segment_lines(img: np.ndarray) -> list[tuple[Image.Image, list[int], dict]]:
    """Segments the image into individual lines using projection profiles and density filtering."""
    print("DEBUG: Executing Line-Guided Segmentation (Iteration 3)")
    # 0. Deskew first
    img = deskew(img)
    
    # 1. Prepare threshold for line detection
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    # 2. Detect ruled lines
    ruled_y = detect_ruled_lines(thresh)
    print(f"Detected {len(ruled_y)} ruled guide lines.")
    
    # 3. Clean threshold for text detection (remove lines but keep for guidance)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w_img // 15), 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h_img // 15)))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    

    
    # LESS aggressive dilation for line removal to preserve text
    horiz_mask = cv2.dilate(horiz_lines, np.ones((2, 1), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert_lines, np.ones((1, 2), np.uint8), iterations=1)
    thresh_clean = cv2.subtract(thresh, horiz_mask)
    thresh_clean = cv2.subtract(thresh_clean, vert_mask)
    
    # 4. Use ruled lines as anchors if available (require at least 3 to trust them)
    line_images = []
    if len(ruled_y) >= 3:
        # Calculate median distance if possible
        distances = [ruled_y[i] - ruled_y[i-1] for i in range(1, len(ruled_y))]
        median_dist = int(np.median(distances))
            
        padding_above = int(median_dist * 0.8) # Slightly more top padding
        padding_below = int(median_dist * 0.2) # Less bottom padding to avoid next line
        print(f"Median line distance: {median_dist}px, Above: {padding_above}px, Below: {padding_below}px")
        
        for i, y_anchor in enumerate(ruled_y):
            y_low = max(0, y_anchor - padding_above)
            y_high = min(h_img, y_anchor + padding_below)
            
            # Increase threshold to avoid empty lines with marginal noise/ruling remnants
            roi_thresh = thresh_clean[y_low:y_high, :]
            density = np.count_nonzero(roi_thresh) / roi_thresh.size if roi_thresh.size > 0 else 0
            
            meta = {"density": density, "rejected": False, "reason": ""}
            if density <= 0.008: 
                meta["rejected"] = True
                meta["reason"] = f"Density ({density:.4f} < 0.008)"
            else:
                roi_color = img[y_low:y_high, :]
                is_noise, reason = is_noise_segment(roi_color, roi_thresh)
                if is_noise:
                    meta["rejected"] = True
                    meta["reason"] = reason
            
            # We now keep the segment even if rejected, so it can be shown in UI
            roi_color = img[y_low:y_high, :]
            
            # NEW: Tight Cropping based on ink projection
            # 1. Horizontal
            x_start, x_end = robust_horizontal_crop(roi_thresh, breathing_room=15)
            
            # 2. Vertical Refining
            coords_y = np.where(np.any(roi_thresh[:, x_start:x_end] > 0, axis=1))[0]
            if coords_y.size > 0:
                y_start = max(0, y_low + coords_y[0] - 5)
                y_end = min(h_img, y_low + coords_y[-1] + 5)
            else:
                y_start, y_end = y_low, y_high

            roi_color_cropped = img[y_start:y_end, x_start:x_end]
            clean_roi = remove_ruling_lines(roi_color_cropped)
            masked_roi = apply_schwarzmaske(clean_roi, padding_bottom=max(0, y_high - y_end))
            img_line = Image.fromarray(cv2.cvtColor(masked_roi, cv2.COLOR_BGR2RGB))
            
            # bbox: [x, y, width, height]
            line_images.append((img_line, [int(x_start), int(y_start), int(x_end - x_start), int(y_end - y_start)], meta))
    else:
        # FALLBACK: Use projection logic if no/few ruled lines found
        print(f"Ruled lines ({len(ruled_y)}) insufficient. Using projection segmentation.")
        # Projection logic here... (keep existing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
        dilated = cv2.dilate(thresh_clean, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for cnt in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if w_c < 25 or h_c < 5: continue
            roi_thresh = thresh_clean[y_c:y_c+h_c, x_c:x_c+w_c]
            roi_color = img[y_c:y_c+h_c, x_c:x_c+w_c]
            split = split_lines_by_projection(roi_thresh, roi_color)
            # Adjust split coordinates to original image
            for s_img, s_bbox, s_meta in split:
                s_bbox[0] += x_c
                s_bbox[1] += y_c
                line_images.append((s_img, s_bbox, s_meta))
            
    if not line_images:
        print(f"Line-Guided Segmentation: result is EMPTY (all filtered as noise or density too low).")
    else:
        print(f"Line-Guided Segmentation: result has {len(line_images)} lines.")
    return line_images

def pad_image(image: Image.Image, target_size=(384, 384)) -> Image.Image:
    """
    Pads the image to a target aspect ratio (e.g. 4:1) before resizing to square,
    to reduce extreme vertical stretching.
    """
    # Move to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    orig_w, orig_h = image.size
    
    # Target aspect ratio 4:1 (common for handwriting lines)
    target_ratio = 4.0
    current_ratio = orig_w / orig_h
    
    if current_ratio > target_ratio:
        # Too wide: pad vertically to reach 4:1
        new_h = int(orig_w / target_ratio)
        new_image = Image.new("RGB", (orig_w, new_h), (255, 255, 255))
        offset = (0, (new_h - orig_h) // 2)
        new_image.paste(image, offset)
        image = new_image
    elif current_ratio < 0.5:
        # Too tall: pad horizontally
        new_w = int(orig_h * 0.5)
        new_image = Image.new("RGB", (new_w, orig_h), (255, 255, 255))
        offset = ((new_w - orig_w) // 2, 0)
        new_image.paste(image, offset)
        image = new_image

    return image
