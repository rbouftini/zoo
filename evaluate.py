from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset, load_from_disk
import torch
import os
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
model_names = ["SFT", "DPO/checkpoint-5178"]

def collate(batch):
    inputs = [[{"role": "user", "content": x}] for x in batch]
    inputs = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    return {"prompts": batch, "inputs": inputs}

rows = {"prompt": []}
for i, model_name in enumerate(model_names):
    print("Model:", model_name)
    rows[model_name] = []
    path = f"checkpoints/{model_name}"
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto")
    if os.path.isfile(f"{path}/tokenizer.json"):
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        print("No local tokenizer is available for this model. Refering back to it's base version.")
        tokenizer = AutoTokenizer.from_pretrained(f"checkpoints/Qwen3-0.6B-Base")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs[:10]")
    dataset = dataset.filter(lambda ex: len(tokenizer.encode(ex["prompt"])) < 200)
    print("Number of examples in dataset:", dataset.num_rows)
    column_names = dataset.column_names
    column_names.remove("prompt")
    dataset = dataset.remove_columns(column_names=column_names)

    val_loader = DataLoader(dataset=dataset["prompt"], batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)
    for x in val_loader:
        inputs = x["inputs"]
        prompts = x["prompts"]
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512)
            for ins, outs, prompt  in zip(inputs["input_ids"], output_ids, prompts):
                answer = tokenizer.decode(outs[ins.shape[0]:], skip_special_tokens=True)
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                rows[model_name].append(messages)
                if i == 0:
                    rows["prompt"].append(prompt)

dataset = Dataset.from_dict(rows)
dataset.save_to_disk("../SFTvsDPO")

model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
batch_size = 2

rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map=device,
    dtype = torch.bfloat16,
    num_labels=1,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def collate(batch):
    batch = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=False)
    batch = tokenizer(batch, return_tensors="pt", padding=True).to(device)
    return batch

n_examples = dataset.num_rows

sft_loader = DataLoader(dataset["SFT"], batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)
dpo_loader = DataLoader(dataset["DPO"], batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

n_w = 0
for a_0, a_1 in zip(sft_loader, dpo_loader):
   time_start = time.time()
   with torch.no_grad():
    scores_0 = rm(**a_0).logits
    scores_1 = rm(**a_1).logits
    n_w += (scores_1 > scores_0).long().sum(0).item()

   time_end = time.time()
   print(f"Time taken: {(time_end - time_start):.4f}")

acc = n_w/ n_examples
print(f"DPO's winrate vs SFT: {acc:.4f}")