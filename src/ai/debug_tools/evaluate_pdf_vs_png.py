import os
import io
import json
import torch
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

# Model settings
MODEL_ID = "fhswf/TrOCR_german_handwritten" # Use German as in the metadata
# Note: The user said 'en' adapted model was used too, but let's stick to what's active.
# Actually, Lina Wegner is German, but the text is English ("Four generations...").
# The user metadata says it's English "English klausur".

def get_model(lang):
    if lang == "de":
        model_name = "fhswf/TrOCR_german_handwritten"
    else:
        model_name = "microsoft/trocr-base-handwritten"
    
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

def calculate_cer(gt, pred):
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    if not gt: return 0.0
    return edit_distance(gt, pred) / len(gt)

# Paths
base_dir = "/home/t68/eclipse-workspace/flAICheckBot/src/test/resources/local-test-data"
pdf_path = os.path.join(base_dir, "20251226-1035.pdf")
png_path = os.path.join(base_dir, "test_handwriting_real.png")
gt_path = os.path.join(base_dir, "test_handwriting_real.txt")

# Load GT
with open(gt_path, 'r') as f:
    gt_text = f.read().strip()

# Initialize AI Components (Mocking the backend logic)
# We'll use the same logic as in icr_prototype.py
device = "cuda" if torch.cuda.is_available() else "cpu"
processor, model = get_model("de") # Using German base for this student? Or EN?
# The user said he improved recognition with and without training.
# Let's use the one that's likely active.
model.to(device)

def run_ocr(img_np):
    # This simulates the backend pipeline: segmentation -> ocr
    # For simplicity in evaluation, we'll assume we want to see the whole page or 
    # we'll use the backend's segment_lines if possible.
    
    from app.preprocessing import segment_lines, advanced_preprocess
    
    results = []
    line_results = segment_lines(img_np)
    for line_img, bbox, _ in line_results:
        line_np = cv2.cvtColor(np.array(line_img), cv2.COLOR_RGB2BGR)
        processed_img_np = advanced_preprocess(line_np)
        processed_image = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
        
        pixel_values = processor(images=processed_image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(pixel_values=pixel_values)
        
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        results.append(text)
    
    return "\n".join(results)

# 1. Evaluate PNG
print("Evaluating PNG...")
img_png = cv2.imread(png_path)
pred_png = run_ocr(img_png)
cer_png = calculate_cer(gt_text, pred_png)
print(f"PNG CER: {cer_png:.4f}")

# 2. Evaluate PDF at different DPIs
dpis = [96, 150, 200, 300]
for dpi in dpis:
    print(f"\nEvaluating PDF at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)
    img_pdf = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    pred_pdf = run_ocr(img_pdf)
    cer_pdf = calculate_cer(gt_text, pred_pdf)
    print(f"PDF ({dpi} DPI) CER: {cer_pdf:.4f}")
    
    # Save predictions for diff
    with open(f"pred_pdf_{dpi}.txt", "w") as f:
        f.write(pred_pdf)

with open("pred_png.txt", "w") as f:
    f.write(pred_png)
