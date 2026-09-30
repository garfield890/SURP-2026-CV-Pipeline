from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_path = "./models/qwen2.5-0.5b"

print("Loading Qwen2.5-0.5B-Instruct model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Example array of bounding box labels and confidence scores
detections = [
    ("person", 0.94),
    ("laptop", 0.88),
    ("chair", 0.91),
    ("coffee cup", 0.85)
]

# Format detections into a clear string
det_str = ", ".join([f"{cls} ({conf*100:.1f}%)" for cls, conf in detections])

messages = [
    {
        "role": "system", 
        "content": "You are a concise scene descriptor. Given a list of detected object classes and confidence scores, describe the scene in one natural sentence. Do not mention percentages, confidence scores, or numerical values."
    },
    {
        "role": "user", 
        "content": f"Detections: {det_str}"
    }
]

# Time tokenization and prompt preparation
t_prep_start = time.perf_counter()
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
t_prep = (time.perf_counter() - t_prep_start) * 1000

# Synchronize GPU/MPS before timing inference
if model.device.type == "mps":
    torch.mps.synchronize()
elif model.device.type == "cuda":
    torch.cuda.synchronize()

t_infer_start = time.perf_counter()

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=60,
    temperature=0.7,
    do_sample=True
)

# Synchronize GPU/MPS after generation
if model.device.type == "mps":
    torch.mps.synchronize()
elif model.device.type == "cuda":
    torch.cuda.synchronize()

t_infer_end = time.perf_counter()
infer_duration = t_infer_end - t_infer_start

# Compute generated tokens and speed
num_input_tokens = model_inputs.input_ids.shape[1]
num_total_tokens = generated_ids.shape[1]
num_new_tokens = num_total_tokens - num_input_tokens
tokens_per_sec = num_new_tokens / infer_duration if infer_duration > 0 else 0

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print("\n--- Detections Input ---")
print(det_str)
print("\n--- Generated Scene Description ---")
print(response)

print("\n--- Telemetry & Performance ---")
print(f"Tokenization / Prompt Prep   : {t_prep:.1f} ms")
print(f"Model Inference Time         : {infer_duration * 1000:.1f} ms ({infer_duration:.2f} s)")
print(f"Tokens Generated             : {num_new_tokens} tokens")
print(f"Generation Speed             : {tokens_per_sec:.1f} tokens/sec")

