import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image

# Mock the functions to avoid importing icr_prototype (which loads models)
def deskew(img):
    # Just return as is for debug or implement a simple version
    return img

def segment_lines(img):
    # We can't easily mock this without importing, so let's import it BUT 
    # make sure icr_prototype doesn't load everything on import if possible.
    # Since I can't easily change icr_prototype's top-level, I'll just 
    # try to use cv2 logic directly or hope for the best.
    
    # Actually, let's just use the functions from icr_prototype but 
    # wrap the import.
    try:
        from icr_prototype import segment_lines, deskew
        return segment_lines(img), deskew
    except Exception as e:
        print(f"Import failed: {e}")
        return [], None

# Paths
base_dir = "/home/t68/eclipse-workspace/flAICheckBot/src/test/resources/local-test-data"
pdf_path = os.path.join(base_dir, "20251226-1035.pdf")

print("Rendering PDF at 200 DPI...")
images = convert_from_path(pdf_path, dpi=200)
img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

print("Running segmentation...")
line_info, deskew_func = segment_lines(img)

# Draw segmentation on the image
debug_img = img.copy()
for i, (line_img, bbox) in enumerate(line_info):
    x, y, w, h = bbox
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(debug_img, str(i), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("debug_pdf_page_1_segmentation.png", debug_img)
print("Saved debug_pdf_page_1_segmentation.png")

if len(line_info) > 5:
    l_img, l_bbox = line_info[5]
    l_np = cv2.cvtColor(np.array(l_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_line_5_raw.png", l_np)
    
    # Deskewed
    if deskew_func:
        l_deskewed = deskew_func(l_np)
        cv2.imwrite("debug_line_5_deskewed.png", l_deskewed)
    print("Saved line 5 samples")

