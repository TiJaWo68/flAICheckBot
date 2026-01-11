import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
import os
import cv2
import numpy as np
from PIL import Image
import re
import whisper
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

from .config import LANGUAGE_BASE_MODELS, DEFAULT_BASE_MODEL, ADAPTER_BASE_DIR, PROJECT_ID, LOCATION, VERTEX_MODEL_NAME
from .preprocessing import advanced_preprocess

# Global caches
_processor_cache = {}
_base_model_cache = {}
adapter_cache = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Vertex AI Client
try:
    vertex_client = genai.Client(project=PROJECT_ID, location=LOCATION, vertexai=True)
    print(f"Vertex AI (google-genai) initialized (Project: {PROJECT_ID}, Location: {LOCATION}, Model: {VERTEX_MODEL_NAME})")
except Exception as e:
    print(f"Warning: Failed to initialize Vertex AI Client: {e}")
    vertex_client = None

# Load Whisper model eagerly
print("Loading Whisper model...")
try:
    stt_model = whisper.load_model("base", device=device)
except Exception as e:
    print(f"CRITICAL: Failed to load Whisper model: {e}")
    stt_model = None


def get_device_info():
    """Returns a dict with friendly device name and type."""
    info = {
        "type": "CPU",
        "name": "Generic CPU",
        "icon": "cpu"
    }
    
    try:
        if torch.cuda.is_available():
            info["type"] = "CUDA"
            info["name"] = torch.cuda.get_device_name(0)
            info["icon"] = "cuda"
        elif torch.backends.mps.is_available():
             info["type"] = "MPS"
             info["name"] = "Apple Silicon"
             info["icon"] = "cpu"
        else:
            # CPU detection
            import platform
            proc_info = platform.processor()
            info["name"] = proc_info if proc_info else "Standard CPU"
            
            # Icon heuristic
            lower_name = info["name"].lower()
            if "amd" in lower_name or "ryzen" in lower_name:
                info["icon"] = "amd"
            elif "intel" in lower_name or "core" in lower_name or "xeon" in lower_name:
                info["icon"] = "intel"
                
    except Exception as e:
        print(f"Error checking device info: {e}")
        
    return info

def list_available_models(token: str = "", projectId: str = "", apiKey: str = ""):
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

def get_base_model_and_processor(lang: str):
    """Loads and caches the appropriate base model and processor for a language."""
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
                return get_base_model_and_processor("en")
            raise e
            
    return _base_model_cache[repo_id], _processor_cache[repo_id]

def get_model_and_processor(language: str = "de"):
    global adapter_cache, ADAPTER_BASE_DIR
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

def is_garbage(text: str) -> bool:
    """Heuristic to detect OCR hallucinations from line remnants."""
    txt = text.strip()
    if not txt:
        return True
    
    # 0. Allow common list markers
    if re.match(r'^[\(]?[0-9a-zA-Z]{1,2}[\.\)]\s*$', txt):
        return False

    # 1. Repetitive nonsense numbers
    if re.match(r'^[\s01.-]+$', txt):
        if len(txt) < 8: 
             if txt.strip() in ("0", "0.", "0.0", "-"):
                 return True
             if len(txt.strip()) > 3: 
                 return True
            
    # 3. Long sequences of meaningless digits/symbols or excessive repetition
    if re.search(r'000\d{5,}', txt) or "0000" in txt:
        return True
    
    # 4. Excessive repetition of same char sequences
    words = txt.split()
    if len(words) > 5:
        from collections import Counter
        counts = Counter(words)
        if any(count > 3 for count in counts.values()) and len(counts) < len(words) / 2:
            return True
        
    return False

def predict_line(img_np: np.ndarray, model, processor) -> str:
    """Runs inference on a single line image numpy array."""
    # Preprocessing: Denoising and Deskewing. 
    processed_img_np = advanced_preprocess(img_np)
    processed_image = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
    
    pixel_values = processor(images=processed_image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            repetition_penalty=1.2
        )
    
    generated_ids = outputs.sequences
    line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return line_text, outputs.sequences_scores

# Pre-load default model
print("Pre-loading default model...")
get_base_model_and_processor("de")
