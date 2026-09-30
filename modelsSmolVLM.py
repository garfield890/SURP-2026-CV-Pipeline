import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_dir = "./models/smolvlm2-500m"
model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

os.makedirs(model_dir, exist_ok=True)

print(f"Downloading {model_name}...")

# Download model weights and processor
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(model_name)

# Save locally to project models directory
print(f"Saving to {model_dir}...")
model.save_pretrained(model_dir)
processor.save_pretrained(model_dir)

print(f"\nModel and processor successfully saved locally to {model_dir}")
