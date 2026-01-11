import torch
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import icr_prototype
import os
import shutil

# Use same model as prototype
BASE_MODEL = "microsoft/trocr-large-handwritten"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model and processor...")
processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL).to(device)
model.eval()

def test_on_user_image(image_path, output_dir="debug_lines"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
        
    print("Segmenting lines...")
    line_images = icr_prototype.segment_lines(img)
    print(f"Detected {len(line_images)} lines.")
    
    results = []
    for i, line_img in enumerate(line_images):
        # Save snippet for visual check
        line_img.save(f"{output_dir}/line_{i:02d}.png")
        
        # Test WITHOUT manual padding (let processor handle it)
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=pixel_values)
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Line {i}: {text}")
        results.append(text)
        
    return "\n".join(results)

image_path = "/home/t68/.gemini/antigravity/brain/f4a8f8cd-58e5-460a-ab57-9763a9f72929/uploaded_image_1767807161120.png"
final_text = test_on_user_image(image_path)
print("\n--- FINAL RECOGNITION (NO MANUAL PADDING) ---")
print(final_text)
