from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch
import time

device = "cuda:3" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)
model_name = "SFT"
path = f"checkpoints/{model_name}"

init_seed = 42
batch_size = 4
num_trajectories = 2
mu = 1e-3

rm_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_name,
    device_map=device,
    dtype = torch.bfloat16,
    num_labels=1,
)
tok_rm = AutoTokenizer.from_pretrained(rm_name)

def collate_model(batch):
    inputs = [[{"role": "user", "content": x}] for x in batch]
    inputs = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    return {"prompts": batch, "inputs": inputs}

def collate_rm(batch):
    batch = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=False)
    batch = tokenizer(batch, return_tensors="pt", padding=True)
    return batch

model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path)

model.to(device)

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs") 
dataset = dataset.filter(lambda ex: len(tokenizer.encode(ex["prompt"])) < 300, num_proc=16)
print("Number of examples in dataset:", dataset.num_rows)

train_loader = DataLoader(dataset=dataset["prompt"], batch_size=batch_size, collate_fn=collate_model,
                             shuffle=False, pin_memory=True)

# \theta + \mu * v
def perturb_params(model, mu, seed):
  g = torch.Generator(device=device).manual_seed(seed)
  with torch.no_grad():
    for param in model.parameters():
      v = torch.randn(size=param.shape, device=device, generator=g)
      param.add_(v.to(param.dtype), alpha=mu)


for idx ,x in enumerate(train_loader):
    inputs = x["inputs"]
    prompts = x["prompts"]
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    rows = {"prompt": []}
    seed = init_seed + idx
    for i in range(2):
        if i == 1:
           perturb_params(model, mu, seed)

        rows[str(i)] = []
        with torch.inference_mode():
            outputs_ids = model.generate(**inputs, max_new_tokens=512, num_return_sequences=num_trajectories)
        for j, (ins, prompt) in enumerate(zip(inputs["input_ids"], prompts)):
            for k in range(j*num_trajectories, num_trajectories*(j+1)):
                outs = outputs_ids[k]
                answer = tokenizer.decode(outs[ins.shape[0]:], skip_special_tokens=True)
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                rows[str(i)].append(messages)
                if i == 0:
                    rows["prompt"].append(prompt)

    trajectories = Dataset.from_dict(rows)
    n_trajs = trajectories.num_rows

    normal_loader = DataLoader(trajectories["0"], batch_size=batch_size, collate_fn=collate_rm, shuffle=False, pin_memory=True)
    perturb_loader = DataLoader(trajectories["1"], batch_size=batch_size, collate_fn=collate_rm, shuffle=False, pin_memory=True)

    n_w = 0
    for a_0, a_1 in zip(normal_loader, perturb_loader):
        a_0 = {k: v.to(device, non_blocking=True) for k, v in a_0.items()} 
        a_1 = {k: v.to(device, non_blocking=True) for k, v in a_1.items()} 
        time_start = time.time()
        with torch.no_grad():
            scores_0 = rm(**a_0).logits
            scores_1 = rm(**a_1).logits
            n_w += (scores_1 > scores_0).long().sum(0).item()

        torch.cuda.synchronize() 
        time_end = time.time()
        print(f"Time taken: {(time_end - time_start):.4f}")

    acc = n_w/ n_trajs
    print("Accuracy:", acc)
    if acc <= 1/2:
        perturb_params(model, -2*mu, seed)