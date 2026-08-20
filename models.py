from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./models/qwen2.5-0.5b"

# Download and save weights directly into project folder
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Model saved locally to {model_dir}")
