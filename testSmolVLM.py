from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import time

model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to("mps")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Can you describe this image in one natural sentence?"},
        ]
    },
]

# Time preprocessing & tokenization
t_prep_start = time.perf_counter()
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.float16)
t_prep = (time.perf_counter() - t_prep_start) * 1000

# Synchronize MPS before timing inference
if torch.backends.mps.is_available():
    torch.mps.synchronize()

t_infer_start = time.perf_counter()

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

# Synchronize MPS so all GPU operations complete before recording end time
if torch.backends.mps.is_available():
    torch.mps.synchronize()

t_infer_end = time.perf_counter()
infer_duration = t_infer_end - t_infer_start

# Compute generated tokens and speed
num_input_tokens = inputs["input_ids"].shape[1]
num_total_tokens = generated_ids.shape[1]
num_new_tokens = num_total_tokens - num_input_tokens
tokens_per_sec = num_new_tokens / infer_duration if infer_duration > 0 else 0

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print("\n--- Output ---")
print(generated_texts[0])

print("\n--- Telemetry & Performance ---")
print(f"Preprocessing / Tokenization : {t_prep:.1f} ms")
print(f"Model Inference Time         : {infer_duration * 1000:.1f} ms ({infer_duration:.2f} s)")
print(f"Tokens Generated             : {num_new_tokens} tokens")
print(f"Generation Speed             : {tokens_per_sec:.1f} tokens/sec")
