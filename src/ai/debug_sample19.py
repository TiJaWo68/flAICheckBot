
import os
import sys
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append("/home/t68/eclipse-workspace/flAICheckBot/src/ai")
import icr_prototype
print(f"DEBUG: icr_prototype loaded from {icr_prototype.__file__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_sample19():
    print("Debugging Sample 19 Hallucinations...")
    
    # Load Base Model
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    
    # Config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    img_path = "/home/t68/eclipse-workspace/flAICheckBot/exported_samples/en/sample_19.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(img_path)
    
    # Segment
    # We want to catch the line that produces the hallucination.
    # We'll save the images to inspect what they look like.
    if os.path.exists("debug_19"):
        import shutil
        shutil.rmtree("debug_19")
    os.makedirs("debug_19")
    
    lines = icr_prototype.segment_lines(img)
    print(f"Detected {len(lines)} lines.\n")
    
    print("--- OCR Results ---")
    for i, line_img in enumerate(lines):
        line_path = f"debug_19/line_{i:03d}.png"
        line_img.save(line_path)
        
        processed_img_np = icr_prototype.deskew(cv2.cvtColor(np.array(line_img), cv2.COLOR_RGB2BGR))
        processed_pil = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
        
        pixel_values = processor(images=processed_pil, return_tensors="pt").pixel_values.to(device)
        
        generated_ids = model.generate(
            pixel_values=pixel_values,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            repetition_penalty=1.2,
            max_new_tokens=64
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Check density again manually for debug log
        roi = cv2.imread(line_path)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, ink_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        density = cv2.countNonZero(ink_mask) / (roi.shape[0]*roi.shape[1])
        
        print(f"Line {i} (Dens={density:.4f}): {text}")

if __name__ == "__main__":
    debug_sample19()
