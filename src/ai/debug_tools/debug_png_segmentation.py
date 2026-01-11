import cv2
import numpy as np
import os
from PIL import Image
from icr_prototype import segment_lines, deskew

# Paths
base_dir = "/home/t68/eclipse-workspace/flAICheckBot/src/test/resources/local-test-data"
png_path = os.path.join(base_dir, "test_handwriting_real.png")

img = cv2.imread(png_path)
if img is None:
    print(f"Failed to load {png_path}")
    exit(1)

print(f"PNG size: {img.shape}")

print("Running segmentation on PNG...")
line_info = segment_lines(img)

# Draw segmentation
debug_img = img.copy()
for i, (line_img, bbox) in enumerate(line_info):
    x, y, w, h = bbox
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(debug_img, str(i), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite("debug_png_segmentation.png", debug_img)

if len(line_info) > 5:
    l_img, l_bbox = line_info[5]
    l_np = cv2.cvtColor(np.array(l_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_png_line_5.png", l_np)
    print(f"PNG Line 5 size: {l_np.shape}")
