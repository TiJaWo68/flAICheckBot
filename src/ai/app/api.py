from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import StreamingResponse
import uvicorn
import shutil
import os
import tempfile
import json
import asyncio
import base64
import io
from PIL import Image
import numpy as np
import cv2
import torch
from pdf2image import convert_from_bytes
import sqlite3

from .version import __version__
from .config import DB_PATH
from .ocr import (
    get_model_and_processor, 
    list_available_models, 
    get_device_info, 
    is_garbage, 
    stt_model, 
    device,
    adapter_cache  # used in reset
)
from .preprocessing import (
    segment_lines, 
    advanced_preprocess, 
    pad_image
)
from .training import train_model

app = FastAPI(title="ICR Backend", version=__version__)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

print("\n" + "="*50)
print("ICR BACKEND STARTING")
print(f"VERSION: {__version__}")
print("NOISE REJECTION FILTERS: ACTIVE")
print("DPI DEFAULT: 150")
print("="*50 + "\n")

@app.get("/version")
async def get_version():
    return {"version": __version__}

@app.get("/ping")
async def ping():
    d_info = get_device_info()
    # Assuming vertex model name is constant or available in config if needed
    from .config import VERTEX_MODEL_NAME
    return {
        "status": "ok", 
        "model": VERTEX_MODEL_NAME,
        "device": d_info["type"],
        "deviceName": d_info["name"],
        "deviceIcon": d_info["icon"],
        "version": __version__
    }

@app.get("/models")
async def list_models(token: str = "", projectId: str = "", apiKey: str = ""):
    return list_available_models(token, projectId, apiKey)

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form("de"), preprocess: str = Form("true")):
    try:
        # Read content first
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        # Detect PDF
        is_pdf = file.filename.lower().endswith(".pdf") or file.content_type == "application/pdf"
        
        pages = []
        if is_pdf:
            print(f"PDF detected: {file.filename}. Converting to images...")
            try:
                # Convert PDF to images at 150 DPI
                images = convert_from_bytes(content, dpi=150) 
                for img in images:
                    # convert PIL to OpenCV
                    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    pages.append(open_cv_image)
            except Exception as e:
                print(f"Failed to convert PDF: {e}")
                return {"status": "error", "message": f"Could not process PDF: {str(e)}"}
        else:
            nparr = np.frombuffer(content, np.uint8)
            if nparr.size == 0:
                raise HTTPException(status_code=400, detail="Could not extract data from file")
                
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            pages.append(img)

        use_preprocess = preprocess.lower() == "true"
        current_model, active_processor = get_model_and_processor(language)
        current_model.eval()
        
        async def event_generator():
            all_results = []
            page_total = len(pages)
            
            for page_idx, img in enumerate(pages):
                # 1. Line Segmentation
                if use_preprocess:
                    line_results = segment_lines(img)
                    print(f"Page {page_idx+1}/{page_total}: Detected {len(line_results)} lines.")
                else:
                    # Skip segmentation
                    line_results = [(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [0, 0, img.shape[1], img.shape[0]], {})]
                    print(f"Page {page_idx+1}/{page_total}: Preprocessing disabled: processing whole image.")

                if not line_results:
                    print(f"Page {page_idx+1}/{page_total}: No text/lines detected.")
                    continue

                line_total = len(line_results)

                for i, (line_img, bbox, meta) in enumerate(line_results):
                    # Convert to numpy
                    img_np = cv2.cvtColor(np.array(line_img), cv2.COLOR_RGB2BGR)

                    # 1. Check if already rejected by segmentation (noise)
                    is_rejected = meta.get("rejected", False)
                    rejection_reason = meta.get("reason", "")
                    
                    line_text = ""
                    avg_prob = 1.0
                    ink_density = meta.get("density", 0.0)

                    if not is_rejected:
                        # Preprocessing: Denoising and Deskewing. 
                        processed_img_np = advanced_preprocess(img_np)
                        processed_image = Image.fromarray(cv2.cvtColor(processed_img_np, cv2.COLOR_BGR2RGB))
                        
                        # 3. Process
                        pixel_values = active_processor(images=processed_image, return_tensors="pt").pixel_values.to(device)
                        
                        try:
                            with torch.no_grad():
                                outputs = current_model.generate(
                                    pixel_values=pixel_values,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    num_beams=4,
                                    length_penalty=1.0,
                                    early_stopping=True,
                                    repetition_penalty=1.2
                                )
                            
                            generated_ids = outputs.sequences
                            line_text = active_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                            
                            # Confidence filtering
                            if outputs.sequences_scores is not None:
                                avg_prob = torch.exp(outputs.sequences_scores[0]).item()
                            
                            # Garbage Check
                            if is_garbage(line_text):
                                is_rejected = True
                                rejection_reason = "Garbage Text (Low confidence or noise)"
                                # line_text = "" # Keep text for debugging? The UI might clarify it.
                        except Exception as e:
                            print(f"Error processing line {i}: {e}")
                            is_rejected = True
                            rejection_reason = f"Error: {str(e)}"

                    if not is_rejected:
                        pass # Text already streamed below

                    # Encode segment image to base64 for Java UI
                    buffered = io.BytesIO()
                    line_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    # Result Object (Aligned with Java AIEngineClient protocol)
                    res_obj = {
                        "type": "line",
                        "page": page_idx + 1,
                        "line": i + 1,
                        "index": i + 1,    # Map line -> index for Java
                        "total": line_total, # For this page
                        "text": line_text,
                        "bbox": [int(b) for b in bbox], # Ensure standard ints
                        "confidence": float(avg_prob) if not is_rejected else 0.0,
                        "rejected": is_rejected,
                        "reason": rejection_reason,
                        "density": float(ink_density),
                        "image": img_str
                    }
                    all_results.append(res_obj)
                    
                    # Stream NDJSON
                    yield json.dumps(res_obj, cls=NumpyEncoder) + "\n"
                    # Small delay to yield control loop
                    await asyncio.sleep(0.001) 
            
            # Final Summary Event
            summary = {
                "type": "final",
                "status": "done", 
                "total_lines": len(all_results)
            }
            yield json.dumps(summary, cls=NumpyEncoder) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unhandled error in recognize: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/train")
async def train(language: str = Form("de"), data_path: str = Form(None)):
    if not data_path:
        # Fetch from sqlite
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT s.image_data, s.sample_text 
                FROM training_samples s 
                JOIN training_sets t ON s.training_set_id = t.id 
                WHERE t.language = ?
            """, (language,))
            samples = c.fetchall()
            conn.close()
            
            if not samples:
                return {"status": "error", "message": f"No training samples found for {language} in DB"}
            
            print(f"Training on {len(samples)} samples from DB for {language}...")
            # We need to adapt train_model to accept raw data or we need to pass something else.
            # I updated train_model to take samples_data
            result = train_model(language, samples_data=samples)
            return result
        except Exception as e:
             return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "File path training not fully implemented in refactoring yet."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("de")):
    if not stt_model:
         return {"status": "error", "message": "Whisper model not loaded"}
         
    try:
        # Whisper needs a file path or array. 
        # Easier to save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Transcribe
        result = stt_model.transcribe(tmp_path, language=language)
        os.remove(tmp_path)
        
        return {"status": "success", "text": result["text"].strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reset")
async def reset(language: str = Form(None)):
    global adapter_cache
    if language:
        if language in adapter_cache:
            del adapter_cache[language]
            torch.cuda.empty_cache()
            return {"status": "success", "message": f"Reloaded/Cleared cache for {language}"}
    else:
        adapter_cache.clear()
        torch.cuda.empty_cache()
        return {"status": "success", "message": "Cleared all adapter caches"}

@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    # Debug endpoint to return the preprocessed image
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
        
    p_img = advanced_preprocess(img)
    ret, buf = cv2.imencode(".png", p_img)
    return Response(content=buf.tobytes(), media_type="image/png")
