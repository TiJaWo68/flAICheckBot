
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def reproduce():
    img_path = "/home/t68/eclipse-workspace/flAICheckBot/exported_samples/en/sample_20.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # Use the German model to try and reproduce the error
    repo_id = "fhswf/TrOCR_german_handwritten"
    print(f"Loading model {repo_id}...")
    processor = TrOCRProcessor.from_pretrained(repo_id)
    model = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)
    
    # Matching config from icr_prototype.py
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image.")
        return
        
    # Just use one line (or the whole image as one line for reproduction)
    line_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)
    
    print(f"Generation config: {model.generation_config}")

    print("Generating with default settings...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        print(f"Default - Sequence shape: {outputs.sequences.shape}")
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        print("Default - Success!")
    except Exception as e:
        print(f"Default - Failed: {e}")

    print("\nGenerating with num_beams=5...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=5
            )
        
        print(f"Beam=5 - Sequence shape: {outputs.sequences.shape}")
        
        # Testing the FIX logic
        avg_prob = 1.0
        try:
            beam_indices = outputs.beam_indices if hasattr(outputs, "beam_indices") else None
            print(f"Beam indices available: {beam_indices is not None}")
            
            transition_scores = model.compute_transition_scores(
                outputs.sequences, 
                outputs.scores, 
                beam_indices=beam_indices,
                normalize_logits=True
            )
            print("Beam=5 - Success!")
        except Exception as e:
            print(f"Beam=5 - Failed logic calculation but caught: {e}")

    except Exception as e:
        print(f"Beam=5 - Critical Failure: {e}")
        # traceback.print_exc()

    print("\nGenerating with num_beams=2...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=2
            )
        
        print(f"Beam=2 - Sequence shape: {outputs.sequences.shape}")
        
        # Testing the FIX logic
        try:
            beam_indices = outputs.beam_indices if hasattr(outputs, "beam_indices") else None
            transition_scores = model.compute_transition_scores(
                outputs.sequences, 
                outputs.scores, 
                beam_indices=beam_indices,
                normalize_logits=True
            )
            print("Beam=2 - Success!")
        except Exception as e:
             print(f"Beam=2 - Failed logic calculation but caught: {e}")
             
    except Exception as e:
        print(f"Beam=2 - Critical Failure: {e}")

if __name__ == "__main__":
    reproduce()
