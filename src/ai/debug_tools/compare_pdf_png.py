import cv2
import numpy as np
from pdf2image import convert_from_path
import os

# Paths
base_dir = "/home/t68/eclipse-workspace/flAICheckBot/src/test/resources/local-test-data"
pdf_path = os.path.join(base_dir, "20251226-1035.pdf")
png_path = os.path.join(base_dir, "test_handwriting_real.png")

def analyze_image(img, name):
    print(f"--- Analysis for {name} ---")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Mean brightness: {np.mean(img):.2f}")
    print(f"Std dev: {np.std(img):.2f}")
    
    # Check for noise/grain
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Laplacian Variance (Focus/Noise measure): {laplacian:.2f}")
    
    # Save a small patch for visual comparison
    patch = img[200:500, 200:500]
    cv2.imwrite(f"patch_{name}.png", patch)
    print(f"Saved patch to patch_{name}.png")

# 1. Load Original PNG
img_png = cv2.imread(png_path)
analyze_image(img_png, "original_png")

# 2. Render PDF at 150 DPI (Current logic)
print("\nRendering PDF at 150 DPI...")
images_150 = convert_from_path(pdf_path, dpi=150)
img_pdf_150 = cv2.cvtColor(np.array(images_150[0]), cv2.COLOR_RGB2BGR)
analyze_image(img_pdf_150, "pdf_150dpi")

# 3. Try to find the DPI of the PNG by comparing dimensions
# If the PNG is page 1, let's see its size vs PDF page size
from pypdf import PdfReader
reader = PdfReader(pdf_path)
page = reader.pages[0]
width_pts = float(page.mediabox.width)
height_pts = float(page.mediabox.height)
print(f"\nPDF Page size (pts): {width_pts}x{height_pts}")

png_h, png_w = img_png.shape[:2]
guessed_dpi_w = (png_w / width_pts) * 72
guessed_dpi_h = (png_h / height_pts) * 72
print(f"Guessed DPI of PNG (Width): {guessed_dpi_w:.2f}")
print(f"Guessed DPI of PNG (Height): {guessed_dpi_h:.2f}")

# 4. Render PDF at guessed DPI
target_dpi = round(guessed_dpi_w)
print(f"\nRendering PDF at {target_dpi} DPI...")
images_target = convert_from_path(pdf_path, dpi=target_dpi)
img_pdf_target = cv2.cvtColor(np.array(images_target[0]), cv2.COLOR_RGB2BGR)
analyze_image(img_pdf_target, f"pdf_{target_dpi}dpi")
