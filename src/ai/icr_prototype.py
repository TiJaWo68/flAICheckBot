"""
FastAPI server providing ICR (Intelligent Character Recognition) via TrOCR, 
STT (Speech-to-Text) via Whisper, and image preprocessing.

@author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
"""
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import shutil
import os
import whisper
import tempfile
from PIL import Image
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from peft import PeftModel, LoraConfig, get_peft_model
import os
import cv2
import numpy as np
from fastapi import Response, Form
from fastapi.responses import StreamingResponse
import json
import asyncio
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

DB_PATH = os.getenv("DB_PATH", "flaicheckbot.db")
if not os.path.exists(DB_PATH):
    # Try project root if running from src/ai
    parent_db = "../../flaicheckbot.db"
    if os.path.exists(parent_db):
        DB_PATH = parent_db
# language-specific base models as requested by the user
LANGUAGE_BASE_MODELS = {
    "de": "fhswf/TrOCR_german_handwritten",
    "en": "microsoft/trocr-base-handwritten",
    "fr": "agomberto/trocr-base-handwritten-fr",
    "es": "qantev/trocr-base-spanish"
}
DEFAULT_BASE_MODEL = "microsoft/trocr-base-handwritten"
ADAPTER_BASE_DIR = "./models/lora"

app = FastAPI()

# Vertex AI Configuration
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "flaicheckbot-project") # Placeholder
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

try:
    # New SDK initialization
    vertex_model_name = "gemini-2.0-flash"
    vertex_client = genai.Client(project=PROJECT_ID, location=LOCATION, vertexai=True)
    print(f"Vertex AI (google-genai) initialized (Project: {PROJECT_ID}, Location: {LOCATION}, Model: {vertex_model_name})")
except Exception as e:
    print(f"Warning: Failed to initialize Vertex AI Client: {e}")
    vertex_client = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@app.get("/ping")
async def ping():
    return {"status": "ok", "model": vertex_model_name}

@app.get("/models")
async def list_available_models(token: str = "", projectId: str = "", apiKey: str = ""):
    try:
        if apiKey:
            client = genai.Client(api_key=apiKey)
        elif token:
            client = genai.Client(project=projectId, location=LOCATION, vertexai=True, credentials=Credentials(token=token))
        else:
            client = vertex_client
            
        if not client:
            return {"status": "error", "message": "Client not initialized"}
            
        models = [m.name for m in client.models.list()]
        return {"status": "success", "models": models}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Caches for processors and models
_processor_cache = {}
_base_model_cache = {}
adapter_cache = {} # Map lang_code -> PeftModel

def get_base_model_and_processor(lang: str):
    """Loads and caches the appropriate base model and processor for a language."""
    # Use first 2 chars for base model lookup (e.g. "de-test" -> "de")
    base_lang = lang[:2] if lang and len(lang) >= 2 else "en"
    repo_id = LANGUAGE_BASE_MODELS.get(base_lang, DEFAULT_BASE_MODEL)
    
    if repo_id not in _base_model_cache:
        print(f"Loading base model and processor for '{lang}' ({repo_id})...")
        try:
            p = TrOCRProcessor.from_pretrained(repo_id)
            m = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)
            # Configuration for training/generation
            m.config.decoder_start_token_id = p.tokenizer.cls_token_id
            m.config.pad_token_id = p.tokenizer.pad_token_id
            m.config.eos_token_id = p.tokenizer.sep_token_id
            
            _processor_cache[repo_id] = p
            _base_model_cache[repo_id] = m
        except Exception as e:
            print(f"CRITICAL: Failed to load model for {lang}: {e}")
            if repo_id != DEFAULT_BASE_MODEL:
                return get_base_model_and_processor("en") # Fallback to default
            raise e
            
    return _base_model_cache[repo_id], _processor_cache[repo_id]

print("Pre-loading default model...")
get_base_model_and_processor("de") # Eagerly load German as it's common here

print("Loading Whisper model...")
try:
    stt_model = whisper.load_model("base", device=device) # "base" is a good compromise for speed/accuracy
except Exception as e:
    print(f"CRITICAL: Failed to load Whisper model: {e}")

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

def split_lines_by_projection(roi_thresh: np.ndarray, roi_color: np.ndarray) -> list[Image.Image]:
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
                
                # Internal Density Check for the sub-segment
                sub_thresh = roi_thresh[y_low:y_high, :]
                sub_density = cv2.countNonZero(sub_thresh) / (width * (y_high - y_low))
                if sub_density > 0.015: # 1.5% ink minimum for a valid line
                    img_line = Image.fromarray(cv2.cvtColor(line_roi, cv2.COLOR_BGR2RGB))
                    lines.append((img_line, [0, y_low, width, y_high - y_low]))
    
    # Handle trailing segment
    if in_line and height - start_y > 8:
        line_roi = roi_color[start_y:, :]
        img_line = Image.fromarray(cv2.cvtColor(line_roi, cv2.COLOR_BGR2RGB))
        lines.append((img_line, [0, start_y, width, height - start_y]))
        
    return lines

def segment_lines(img: np.ndarray) -> list[Image.Image]:
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
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 15, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 15))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    
    horiz_mask = cv2.dilate(horiz_lines, np.ones((5, 1), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert_lines, np.ones((1, 5), np.uint8), iterations=1)
    thresh_clean = cv2.subtract(thresh, horiz_mask)
    thresh_clean = cv2.subtract(thresh_clean, vert_mask)
    
    # 4. Use ruled lines as anchors if available
    line_images = []
    if len(ruled_y) >= 2:
        # Calculate median distance
        distances = [ruled_y[i] - ruled_y[i-1] for i in range(1, len(ruled_y))]
        median_dist = int(np.median(distances))
        padding = int(median_dist * 0.25)
        print(f"Median line distance: {median_dist}px, Applying padding: {padding}px")
        
        # Add a synthetic line before the first one if there's space for a header
        # Fix: User says we lose the 1st line. If ruled_y[0] is not at the very top, 
        # always add a virtual line at the top to check for headers.
        if ruled_y[0] > 10:
            ruled_y.insert(0, 5) # Potential line at the very top
            
        # Add a synthetic line after the last one
        if h_img - ruled_y[-1] > median_dist:
            ruled_y.append(ruled_y[-1] + median_dist)
            
        for i in range(len(ruled_y) - 1):
            y_start = ruled_y[i]
            y_end = ruled_y[i+1]
            
            # Extract bucket
            # IMPORTANT: Iteration 5: User wants 0% top, 25% bottom padding
            y_low = y_start
            y_high = min(h_img, y_end + padding)
            
            # Check if this bucket contains text
            roi_thresh = thresh_clean[y_low:y_high, :]
            density = cv2.countNonZero(roi_thresh) / (w_img * (y_high - y_low))
            
            if density > 0.01: # 1% ink to be considered a line
                # Crop and apply line removal + Schwarzmaske
                roi_color = img[y_low:y_high, :]
                clean_roi = remove_ruling_lines(roi_color)
                masked_roi = apply_schwarzmaske(clean_roi, padding_bottom=padding)
                img_line = Image.fromarray(cv2.cvtColor(masked_roi, cv2.COLOR_BGR2RGB))
                line_images.append((img_line, [0, y_low, w_img, y_high - y_low]))
    else:
        # FALLBACK: Use projection logic if no ruled lines found (e.g. plain paper)
        print("No ruled lines found. Falling back to projection segmentation.")
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
            for s_img, s_bbox in split:
                s_bbox[0] += x_c
                s_bbox[1] += y_c
                line_images.append((s_img, s_bbox))
            
    print(f"Line-Guided Segmentation: result has {len(line_images)} lines.")
    return line_images

def pad_image(image: Image.Image, target_size=(384, 384)) -> Image.Image:
    """
    Pads the image to be square (or target aspect ratio) without distorting it.
    """
    # Create a white background
    white_bg = Image.new("RGB", target_size, (255, 255, 255))
    
    # Calculate aspect ratios
    orig_w, orig_h = image.size
    ratio = min(target_size[0] / orig_w, target_size[1] / orig_h)
    
    # Resize keeping aspect ratio
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center the image
    offset = ((target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2)
    white_bg.paste(resized_img, offset)
    
    return white_bg

import io

class HandwritingDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_data, text = self.samples[idx]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to numpy for advanced preprocessing
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_img_np = advanced_preprocess(img_np)
        processed_image = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
        
        # Pad before processing
        image_padded = pad_image(processed_image)
        pixel_values = self.processor(image_padded, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()
        # Important for TrOCR: labels must be -100 for padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

def get_model_and_processor(language: str = "de"):
    global adapter_cache, ADAPTER_BASE_DIR
    # Use full language string for adapter isolation (e.g. "de-test")
    lang = (language or "de").strip()
    
    # 1. Get base model and processor first
    model, processor = get_base_model_and_processor(lang)
    
    # 2. Check for LoRA adapter
    if lang in adapter_cache:
        return adapter_cache[lang], processor
    
    adapter_path = os.path.join(ADAPTER_BASE_DIR, lang)
    if os.path.exists(adapter_path):
        print(f"Loading LoRA adapter for language '{lang}' from {adapter_path}...")
        try:
            peft_model = PeftModel.from_pretrained(model, adapter_path)
            peft_model = peft_model.to(device)
            adapter_cache[lang] = peft_model
            return peft_model, processor
        except Exception as e:
            print(f"Warning: Failed to load adapter for '{lang}': {e}")
            return model, processor
            
    return model, processor

import re

def is_garbage(text: str) -> bool:
    """Heuristic to detect OCR hallucinations from line remnants."""
    txt = text.strip()
    if not txt:
        return True
    
    # 1. Repetitive nonsense numbers like "0 0", "1 1 1", "0 0 0 0"
    if re.match(r'^[\s01.-]+$', txt):
        if len(txt) < 8: # Keep slightly longer numeric sequences if they might be dates
            return True
            
    # 2. Common ruled-line hallucinations
    if "1961 62" in txt and len(txt) < 12:
        return True
    
    # 3. Long sequences of meaningless digits/symbols
    if re.search(r'000\d{5,}', txt) or "0000" in txt:
        return True
        
    return False

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form("de"), preprocess: str = Form("true")):
    try:
        # Read content first to avoid SpooledTemporaryFile issues
        content = await file.read()
        if not content:
            return {"status": "error", "message": "Uploaded file is empty"}
            
        nparr = np.frombuffer(content, np.uint8)
        if nparr.size == 0:
            return {"status": "error", "message": "Could not extract data from file"}
            
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Could not decode image"}

        use_preprocess = preprocess.lower() == "true"
        
        # 1. Line Segmentation (Internal logic handles deskewing and cleaning)
        if use_preprocess:
            line_results = segment_lines(img)
            print(f"Detected {len(line_results)} lines.")
        else:
            # Skip segmentation: treat entire image as one "line"
            line_results = [(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [0, 0, img.shape[1], img.shape[0]])]
            print("Preprocessing disabled: processing whole image.")

        if not line_results:
            # Fallback for single line/small area
            line_results = [(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [0, 0, img.shape[1], img.shape[0]])]

        current_model, active_processor = get_model_and_processor(language)
        current_model.eval()
        
        async def event_generator():
            results = []
            total = len(line_results)
            for i, (line_img, bbox) in enumerate(line_results):
                # Convert to numpy
                img_np = cv2.cvtColor(np.array(line_img), cv2.COLOR_RGB2BGR)
                
                # Preprocessing: Deskew only. 
                processed_img_np = deskew(img_np)
                processed_image = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
                
                # 3. Process
                pixel_values = active_processor(images=processed_image, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    outputs = current_model.generate(
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        repetition_penalty=1.2
                    )
                
                generated_ids = outputs.sequences
                line_text = active_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Confidence filtering
                avg_prob = 1.0 
                try:
                    beam_indices = outputs.beam_indices if hasattr(outputs, "beam_indices") else None
                    transition_scores = current_model.compute_transition_scores(
                        outputs.sequences, 
                        outputs.scores, 
                        beam_indices=beam_indices,
                        normalize_logits=True
                    )
                    avg_prob = torch.exp(transition_scores).mean().item()
                except Exception as e:
                    avg_prob = 1.0 
                
                # Pattern rejection
                if line_text:
                    if is_garbage(line_text):
                        print(f"Filtering garbage pattern: '{line_text}'")
                        continue
                    
                    threshold = 0.35 if len(line_text) > 15 else 0.5
                    if avg_prob < threshold:
                        print(f"Skipping low confidence line ({avg_prob:.2f}): '{line_text}'")
                        continue
                        
                    results.append(line_text)
                    
                    # Yield progress update
                    yield json.dumps({
                        "type": "line",
                        "index": i,
                        "total": total,
                        "text": line_text,
                        "bbox": bbox
                    }) + "\n"
            
            # Yield final result
            final_text = "\n".join(results)
            yield json.dumps({
                "type": "final",
                "status": "success",
                "text": final_text,
                "lines_detected": total
            }) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
    except Exception as e:
        import traceback
        print(f"Error in /recognize: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@app.post("/train")
async def train(language: str = Form("de"), data_path: str = Form(None)):
    try:
        samples = []
        if data_path and os.path.exists(data_path):
            print(f"Loading training data from filesystem: {data_path}")
            # Load images and text files from the directory
            # Expects pairs like name.png and name.txt
            valid_extensions = {".png", ".jpg", ".jpeg"}
            for filename in os.listdir(data_path):
                base, ext = os.path.splitext(filename)
                if ext.lower() in valid_extensions:
                    img_path = os.path.join(data_path, filename)
                    txt_path = os.path.join(data_path, base + ".txt")
                    
                    if os.path.exists(txt_path):
                        try:
                            with open(img_path, "rb") as f:
                                img_data = f.read()
                            with open(txt_path, "r", encoding="utf-8") as f:
                                text_data = f.read().strip()
                            
                            if text_data:
                                samples.append((img_data, text_data))
                        except Exception as e:
                            print(f"Skipping {filename}: {e}")
        
        # Fallback to DB if no samples loaded from path
        if not samples:
            # 1. Fetch data from SQLite
            if not os.path.exists(DB_PATH):
                return {"status": "error", "message": f"Database file not found at {os.path.abspath(DB_PATH)}"}
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT s.image_data, COALESCE(s.sample_text, t.original_text) as text 
                FROM training_samples s
                JOIN training_sets t ON s.training_set_id = t.id
                WHERE t.language = ?
            """, (language,))
            samples = cursor.fetchall()
        finally:
            conn.close()

        if not samples:
            return {"status": "error", "message": "No training samples found in database."}

        print(f"Starting training with {len(samples)} samples...")

        # 2. Prepare PEFT/LoRA
        # Fetch the appropriate base model and processor for this language
        # Use full language string for the target adapter
        # But base model lookup inside get_base_model_and_processor handles the mapping to "de", "en" etc.
        base_model_for_lang, active_processor = get_base_model_and_processor(language)
        
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        
        lora_model = get_peft_model(base_model_for_lang, config)
        
        # 3. Training Loop
        dataset = HandwritingDataset(samples, active_processor)
        
        output_dir = f"./trocr-checkpoint-{language}"
        save_path = os.path.join(ADAPTER_BASE_DIR, language)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1, # Reduce batch size further for reliability
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            logging_steps=5,
            save_strategy="no",
            prediction_loss_only=True,
            remove_unused_columns=False
        )

        trainer = Seq2SeqTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )
        trainer.train()
        
        # 5. Save the trained adapter
        os.makedirs(ADAPTER_BASE_DIR, exist_ok=True)
        lora_model.save_pretrained(save_path)
        print(f"Training complete. Adapter saved to {save_path}")
        
        # Invalidate cache for this language
        if language in adapter_cache:
            del adapter_cache[language]
        
        return {"status": "success", "message": f"Training completed for language '{language}'. Model saved."}
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in /train:\n{error_msg}")
        return {"status": "error", "message": str(e), "trace": error_msg}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("de")):
    try:
        # Save uploaded file to temp with correct extension
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Read content first to avoid SpooledTemporaryFile issues
            content = await file.read()
            if not content:
                return {"status": "error", "message": "Uploaded audio file is empty"}
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            print(f"Transcribing audio ({language}): {tmp_path}")
            # Use the provided language
            result = stt_model.transcribe(tmp_path, language=language)
            transcription = result["text"].strip()
            return {"status": "success", "text": transcription}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Error in /transcribe: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/reset")
async def reset(language: str = Form(None)):
    try:
        global adapter_cache
        if language:
            lang = language.strip()
            if lang in adapter_cache:
                del adapter_cache[lang]
            adapter_path = os.path.join(ADAPTER_BASE_DIR, lang)
            if os.path.exists(adapter_path):
                shutil.rmtree(adapter_path)
                return {"status": "success", "message": f"Adapter for language '{lang}' deleted."}
            else:
                return {"status": "success", "message": f"No adapter found for language '{lang}'."}
        else:
            # Reset all
            adapter_cache = {}
            if os.path.exists(ADAPTER_BASE_DIR):
                shutil.rmtree(ADAPTER_BASE_DIR)
            return {"status": "success", "message": "All language-specific adapters deleted."}
    except Exception as e:
        print(f"Error in /reset: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    try:
        # 1. Read image
        image_bytes = await file.read()
        if not image_bytes:
            return {"status": "error", "message": "Uploaded image for preprocessing is empty"}
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        if nparr.size == 0:
            return {"status": "error", "message": "Could not extract image data"}
            
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Could not decode image"}

        # 2. Convert to gray and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use Otsu's thresholding to isolate text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Deskewing
        # Find all foreground pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 50: # Only if enough points
            # Get the minimum area rectangle that covers all points
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            
            # minAreaRect angle can be tricky. Standardize it.
            # In newer OpenCV, angle is [0, 90]. In older, [-90, 0).
            # We want to find the rotation needed to make it horizontal.
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle
            
            # Limit correction to +/- 15 degrees to avoid massive rotations (like sideways)
            if abs(angle) < 15:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                print(f"Skipping large rotation attempt: {angle:.2f} degrees")

        # 4. Optional: Perspective / Trap correction (if we find a big rect)
        # For now, let's keep it robust with just deskewing + contrast
        
        # Enhancement: Adaptive Thresholding for better visibility (optional)
        # img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

        # 5. Encode back
        _, enc = cv2.imencode(".png", img)
        return Response(content=enc.tobytes(), media_type="image/png")
        
    except Exception as e:
        print(f"Error in /preprocess: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/grade")
async def grade(task: str = Form(...), expected: str = Form(""), actual: str = Form(...)):
    try:
        # Mocking AI grading logic for the prototype
        # In a production system, this could be a call to a local LLM or a scoring algorithm
        
        feedback = f"Bewertung für die Aufgabe: {task[:50]}...\n\n"
        feedback += "Analyse des Schülertextes:\n"
        
        if not actual.strip():
            feedback += "- Keine Antwort erkannt.\n"
            score = 0
        else:
            # Simple heuristic for prototype: count matching words or similar
            words_expected = set(expected.lower().split())
            words_actual = set(actual.lower().split())
            matches = words_expected.intersection(words_actual)
            
            feedback += f"- Der Text enthält {len(words_actual)} Wörter.\n"
            feedback += f"- {len(matches)} Übereinstimmungen mit dem Erwartungshorizont gefunden.\n"
            
            if len(matches) > len(words_expected) * 0.7:
                feedback += "- Sehr gute Übereinstimmung!\n"
                score = 3
            elif len(matches) > len(words_expected) * 0.4:
                feedback += "- Befriedigende Übereinstimmung.\n"
                score = 2
            else:
                feedback += "- Geringe Übereinstimmung.\n"
                score = 1
                
        feedback += f"\nVorgeschlagene Punktzahl: {score}"
        
        return {"status": "success", "feedback": feedback, "score": score}
    except Exception as e:
        print(f"Error in /grade: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/grade_vertex")
async def grade_vertex(task: str = Form(...), expected: str = Form(""), actual: str = Form(...), 
                       token: str = Form(""), projectId: str = Form(""), apiKey: str = Form("")):
    try:
        current_project = projectId if projectId else PROJECT_ID
        model_to_use = "gemini-2.0-flash" # Default/Fallback
        
        if apiKey:
            # AI Studio mode - use gemini-2.0-flash for consistency (1.5-flash was 404)
            current_client = genai.Client(api_key=apiKey)
            model_to_use = "gemini-2.0-flash"
        elif token:
            # Vertex AI mode - use what worked before
            current_client = genai.Client(
                project=current_project, 
                location=LOCATION, 
                vertexai=True,
                credentials=Credentials(token=token)
            )
            model_to_use = "gemini-2.0-flash"
        elif projectId and projectId != PROJECT_ID:
            current_client = genai.Client(project=projectId, location=LOCATION, vertexai=True)
            model_to_use = "gemini-2.0-flash"
        else:
            current_client = vertex_client
            model_to_use = vertex_model_name

        if not current_client:
            return {"status": "error", "message": "Vertex AI not initialized. Check Project ID and Authentication."}

        # Construct the prompt
        prompt = f"""
Du bist ein hochpräziser Korrektur-Assistent für Lehrkräfte. Deine Aufgabe ist die Korrektur eines kompletten Tests, der aus mehreren Einzelaufgaben besteht.

Deine Vorgehensweise:
1. Gehe jede Aufgabe einzeln durch.
2. Vergleiche die Schülerantwort der jeweiligen Aufgabe mit dem zugehörigen Erwartungshorizont.
3. Vergebe Punkte basierend auf der Übereinstimmung, jedoch maximal bis zur definierten Maximalpunktzahl der Aufgabe.
4. Formuliere für jede Aufgabe ein kurzes, sachliches Feedback.
5. Summiere am Ende alle erreichten Punkte zur Gesamtpunktzahl.

Eingabe-Daten (Struktur)
TEST-KONFIGURATION: {task}

SCHÜLER-ANTWORTEN (OCR-Ergebnis): {actual}

Ausgabe-Format (Strukturvorgabe)
Bitte erstelle die Korrektur streng nach diesem Schema:

ERGEBNIS PRO AUFGABE:

Aufgabe [Nummer/Name]:
Erreichte Punkte: [Punkte] / [Maximalpunkte]
Feedback: [Kurze Begründung der Punktvergabe und inhaltliche Rückmeldung]

... (wiederholen für alle Aufgaben)

GESAMTAUSWERTUNG:
Gesamtpunktzahl: [Summe aller erreichten Punkte] / [Gesamtsumme aller Maximalpunkte]
Abschließendes Feedback: [Zusammenfassender Kommentar zur Leistung im gesamten Test]

Antworten Sie ausschließlich im JSON-Format, wobei das obige Schema in das 'feedback' Feld gehört:
{{
    "status": "success",
    "feedback": "Vollständiger Text nach Schema...",
    "score": [Summe der Punkte als Zahl]
}}
"""

        response = current_client.models.generate_content(
            model=model_to_use,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Clean the response text (sometimes Gemini adds ```json ... ```)
        resp_text = response.text.strip()
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:-3].strip()
        elif resp_text.startswith("```"):
            resp_text = resp_text[3:-3].strip()
        
        # Ensure we can parse it as JSON
        result = json.loads(resp_text)
        return result
        
    except Exception as e:
        err_msg = str(e)
        logger_name = "AI-Engine"
        print(f"[{logger_name}] Error in /grade_vertex: {err_msg}")
        
        # Log more details for debugging (without leaking full tokens)
        print(f"[{logger_name}] Request Details - Project: {current_project}, Location: {LOCATION}, Model: {model_to_use}")
        if token:
            print(f"[{logger_name}] OAuth Token detected (prefix: {token[:10]}...)")
        if apiKey:
            print(f"[{logger_name}] API Key detected (prefix: {apiKey[:5]}...)")

        # Robust error extraction
        clean_msg = "Ein unerwarteter KI-Fehler ist aufgetreten."
        error_code = "UNKNOWN"
        
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            error_code = "QUOTA_EXCEEDED"
            import re
            retry_match = re.search(r"retryDelay': '(\d+s)'", err_msg)
            wait_time = retry_match.group(1) if retry_match else "einem Moment"
            clean_msg = f"Kontingent erschöpft (Limit erreicht). Bitte warten Sie ca. {wait_time} vor dem nächsten Versuch."
        elif "NOT_FOUND" in err_msg or "404" in err_msg:
            error_code = "MODEL_NOT_FOUND"
            # Include the SDK error message because it contains valuable "models/..." or project info
            clean_msg = f"Modell '{model_to_use}' nicht verfügbar. Details: {err_msg.split(' {')[0]}"
        elif "PERMISSION_DENIED" in err_msg or "403" in err_msg:
            error_code = "AUTH_ERROR"
            clean_msg = "Zugriff verweigert (403). Prüfen Sie Berechtigungen oder Projekt-Aktivierung."
        elif "INVALID_ARGUMENT" in err_msg or "400" in err_msg:
            error_code = "INVALID_REQUEST"
            clean_msg = "Ungültige Anfrage an die KI (400)."
        else:
            clean_msg = err_msg.split(". {")[0]
            
        return {
            "status": "error", 
            "message": clean_msg, 
            "code": error_code,
            "technical_details": err_msg
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
