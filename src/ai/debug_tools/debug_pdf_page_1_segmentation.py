import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image
from app.preprocessing import segment_lines, deskew

# Paths
base_dir = "/home/t68/eclipse-workspace/flAICheckBot/src/test/resources/local-test-data"
pdf_path = os.path.join(base_dir, "20251226-1035.pdf")

print("Rendering PDF at 200 DPI...")
images = convert_from_path(pdf_path, dpi=200)
img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

print("Running segmentation...")
line_info = segment_lines(img)

# Draw segmentation on the image
debug_img = img.copy()
for i, (line_img, bbox, _) in enumerate(line_info):
    x, y, w, h = bbox
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(debug_img, str(i), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("debug_pdf_page_1_segmentation.png", debug_img)
print("Saved debug_pdf_page_1_segmentation.png")

if len(line_info) > 5:
    l_img, l_bbox, _ = line_info[5]
    l_np = cv2.cvtColor(np.array(l_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_line_5_raw.png", l_np)
    
    # Deskewed
    l_deskewed = deskew(l_np)
    cv2.imwrite("debug_line_5_deskewed.png", l_deskewed)
    print("Saved line 5 samples")

