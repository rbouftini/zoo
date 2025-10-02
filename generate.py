from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen3-0.6B-Base"

print("Before SFT:")
msg = "Hello, how are you?"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(msg, return_tensors="pt").to(device)

output_ids = model.generate(**input_ids, max_new_tokens=200)
answer = tokenizer.decode(output_ids[0])
print(answer)

print("After SFT:")
msg = [{"role":"user", "content": "Hello, how are you?"}]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("checkpoints/checkpoint-250", dtype="auto", device_map="auto")

prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
input_ids = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(input_ids=input_ids["input_ids"], attention_mask= input_ids["attention_mask"], max_new_tokens=200)
answer = tokenizer.decode(output_ids[0])
print(answer)