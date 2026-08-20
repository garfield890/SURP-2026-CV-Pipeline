from transformers import AutoModelForCausalLM, AutoTokenizer

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

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=60,
    temperature=0.7,
    do_sample=True
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print("\n--- Detections Input ---")
print(det_str)
print("\n--- Generated Scene Description ---")
print(response)
