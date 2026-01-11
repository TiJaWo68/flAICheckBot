from torch.utils.data import Dataset
from PIL import Image
import io
import cv2
import numpy as np
from .preprocessing import advanced_preprocess, pad_image
from .ocr import get_model_and_processor
from .config import ADAPTER_BASE_DIR
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import os

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

def train_model(language: str, data_path: str = None, samples_data = None):
    # This function encapsulates the training logic roughly found in the original endpoint
    # Note: The original endpoint had a lot of logic inside the API handler.
    # Ideally, we pass the data here.
    
    model, processor = get_model_and_processor(language)
    
    # If using LoRA
    # Check if already a PeftModel, if not, convert
    from peft import PeftModel
    
    # Ensure model is ready for training
    model.train()
    # Required for some models when using LoRA/gradient checkpointing (even if off)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if not isinstance(model, PeftModel):
        print(f"Converting base model to LoRA for {language}...")
        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query", "value"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, config)
    else:
        # If loading existing adapter, ensure it is trainable
        print(f"Resuming training with existing adapter for {language}...")
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    
    model.print_trainable_parameters()
    
    dataset = HandwritingDataset(samples_data, processor)
    
    # Training args (simplified from original)
    output_dir = os.path.join(ADAPTER_BASE_DIR, language)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        predict_with_generate=True
    )
    
    import torch
    
    def collate_fn(features):
        pixel_values = [item["pixel_values"] for item in features]
        labels = [item["labels"] for item in features]
        
        # Stack images
        pixel_values_stack = torch.stack(pixel_values)
        
        # Pad labels using the tokenizer
        # We wrap labels in a dict as 'input_ids' just for padding logic
        padded = processor.tokenizer.pad(
            {"input_ids": labels}, 
            return_tensors="pt", 
            padding=True
        )
        labels_padded = padded["input_ids"]
        
        # Reset padding to -100 to ignore loss on padding tokens
        labels_padded[labels_padded == processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values_stack, "labels": labels_padded}

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer, 
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    return {"status": "success", "message": f"Training completed for {language}"}
