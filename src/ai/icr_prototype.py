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
import json
from google import genai
from google.genai import types

DB_PATH = "../../flaicheckbot.db"
ADAPTER_PATH = "./trocr-adapter"
BASE_MODEL = "microsoft/trocr-base-handwritten"

app = FastAPI()

# Vertex AI Configuration
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "flaicheckbot-project") # Placeholder
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

try:
    # New SDK initialization
    vertex_client = genai.Client(project=PROJECT_ID, location=LOCATION, vertexai=True)
    vertex_model_name = "gemini-2.0-flash"
    print(f"Vertex AI (google-genai) initialized (Project: {PROJECT_ID}, Location: {LOCATION})")
except Exception as e:
    print(f"Warning: Failed to initialize Vertex AI Client: {e}")
    vertex_client = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "AI Engine is alive"}

print("Loading TrOCR model...")
try:
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    base_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL).to(device)
    
    # Required for training VisionEncoderDecoderModels
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.eos_token_id = processor.tokenizer.sep_token_id
except Exception as e:
    print(f"CRITICAL: Failed to load base model: {e}")

print("Loading Whisper model...")
try:
    stt_model = whisper.load_model("base", device=device) # "base" is a good compromise for speed/accuracy
except Exception as e:
    print(f"CRITICAL: Failed to load Whisper model: {e}")

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
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()
        # Important for TrOCR: labels must be -100 for padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

def get_active_model():
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading trained adapter from {ADAPTER_PATH}...")
        try:
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            return model.to(device)
        except Exception as e:
            print(f"Warning: Failed to load adapter: {e}")
    return base_model

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        current_model = get_active_model()
        current_model.eval()
        
        with torch.no_grad():
            generated_ids = current_model.generate(pixel_values=pixel_values)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"status": "success", "text": generated_text}
    except Exception as e:
        print(f"Error in /recognize: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/train")
async def train():
    try:
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
            """)
            samples = cursor.fetchall()
        finally:
            conn.close()

        if not samples:
            return {"status": "error", "message": "No training samples found in database."}

        print(f"Starting training with {len(samples)} samples...")

        # 2. Prepare PEFT/LoRA
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        
        lora_model = get_peft_model(base_model, config)
        
        # 3. Training Loop
        dataset = HandwritingDataset(samples, processor)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="./trocr-checkpoint",
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
        
        # Save the adapter
        lora_model.save_pretrained(ADAPTER_PATH)
        
        return {"status": "success", "message": f"Training complete with {len(samples)} samples. Adapter saved."}
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
            shutil.copyfileobj(file.file, tmp)
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
async def reset():
    try:
        if os.path.exists(ADAPTER_PATH):
            shutil.rmtree(ADAPTER_PATH)
            print(f"Deleted adapter at {ADAPTER_PATH}")
            return {"status": "success", "message": "AI Training reset. Adapter deleted."}
        else:
            return {"status": "success", "message": "Nothing to reset. Base model is already active."}
    except Exception as e:
        print(f"Error in /reset: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    try:
        # 1. Read image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
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
                       token: str = Form(""), projectId: str = Form("")):
    try:
        current_project = projectId if projectId else PROJECT_ID
        
        if token:
            # Initialize Client with provided token
            current_client = genai.Client(
                project=current_project, 
                location=LOCATION, 
                vertexai=True,
                credentials=types.Credentials(access_token=token)
            )
        elif projectId and projectId != PROJECT_ID:
            # Initialize Client with specific project
            current_client = genai.Client(project=projectId, location=LOCATION, vertexai=True)
        else:
            current_client = vertex_client

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
            model="gemini-2.0-flash",
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
        print(f"Error in /grade_vertex: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
