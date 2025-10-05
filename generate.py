from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen3-0.6B-Base"

print("Before SFT:")
msg = "Hello, how are you doing today?"
model1 = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(msg, return_tensors="pt").to(device)

output_ids = model1.generate(**input_ids, max_new_tokens=200)
answer1 = tokenizer.decode(output_ids[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)

print("After SFT:")
msg = [{"role":"user", "content": "Hello, how are you doing today?"}]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model2 = AutoModelForCausalLM.from_pretrained("checkpoints/checkpoint-250", dtype="auto", device_map="auto")

prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
input_ids = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model2.generate(**input_ids, max_new_tokens=200)
answer2 = tokenizer.decode(output_ids[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
print("User Question:", msg[0]["content"])
print("Model Response:")
print(answer2)

"""
print("RLHF Eval")
import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM", device="cpu")
inputs = [msg[0]["content"]]
candidates_texts = [[answer1, answer2]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=1)
print(inputs)
print(candidates_texts)
print(ranks)
"""
