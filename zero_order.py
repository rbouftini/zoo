from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch
import time
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)
model_name = "SFT"
path = f"checkpoints/{model_name}"

init_seed = 42
batch_size_rm = 8
total_batch_size = 256
batch_size_train = 32
num_trajectories = 4 
accumulation_steps = total_batch_size // (batch_size_train*num_trajectories)
mu = 1e-3
n_epochs = 10
logging_steps = 1

rm_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_name,
    dtype=torch.bfloat16,
    num_labels=1,
).eval()
rm.to(device)

tok_rm = AutoTokenizer.from_pretrained(rm_name)

def collate_model(batch):
    inputs = [[{"role": "user", "content": x}] for x in batch]
    inputs = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    return {"prompts": batch, "inputs": inputs}

def collate_rm(batch):
    batch = tok_rm.apply_chat_template(batch, tokenize=False, add_generation_prompt=False)
    batch = tok_rm(batch, return_tensors="pt", padding=True)
    return batch

model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path)

model.to(device)

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:500]") 
dataset = dataset.filter(lambda ex: len(tokenizer.encode(ex["prompt"])) < 300, num_proc=16)
print("Number of examples in dataset:", dataset.num_rows)

train_loader = DataLoader(dataset=dataset["prompt"], batch_size=batch_size_train, collate_fn=collate_model,
                             shuffle=False, pin_memory=True)

wandb.init(
    project="zero-order-qwen3",
    name=f"qwen3-0.6b-zero-order-{int(time.time())}",
    config={
        "model_name": model_name,
        "train_batch_size": batch_size_train,
        "num_trajectories_per_prompt": num_trajectories,
        "accumulation_steps": accumulation_steps,
        "total_n_trajectories": batch_size_train*num_trajectories,
        "rm_batch_size": batch_size_rm,
        "logging_steps": logging_steps,
        "n_epochs": n_epochs,
        "mu": mu,
        "train_examples": dataset.num_rows,
    },
)

# \theta + \mu * v
def perturb_params(model, mu, seed):
  g = torch.Generator(device=device).manual_seed(seed)
  with torch.no_grad():
    for param in model.parameters():
      v = torch.randn(size=param.shape, device=device, generator=g)
      param.add_(v.to(param.dtype), alpha=mu)

global_step = 0
n_w = 0
n_trajs = 0
mean_reward = 0.0
start_time = time.time()

for epoch in range(n_epochs):
    for micro_step, x in enumerate(train_loader):
        inputs = x["inputs"]
        prompts = x["prompts"]
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        rows = {"prompt": []}
        seed = init_seed + global_step
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
        n_trajs += trajectories.num_rows

        normal_loader = DataLoader(trajectories["0"], batch_size=batch_size_rm, collate_fn=collate_rm, shuffle=False, pin_memory=True)
        perturb_loader = DataLoader(trajectories["1"], batch_size=batch_size_rm, collate_fn=collate_rm, shuffle=False, pin_memory=True)

        for a_0, a_1 in zip(normal_loader, perturb_loader):
            a_0 = {k: v.to(device, non_blocking=True) for k, v in a_0.items()} 
            a_1 = {k: v.to(device, non_blocking=True) for k, v in a_1.items()} 

            with torch.no_grad():
                scores_0 = rm(**a_0).logits
                mean_reward += scores_0.sum(0)
                scores_1 = rm(**a_1).logits
                n_w += (scores_1 > scores_0).long().sum(0).item()
        
        if (micro_step+1) % accumulation_steps == 0:

            torch.cuda.synchronize()
            end_time = time.time()
            time_taken = end_time - start_time

            acc = n_w / n_trajs
            mean_reward = (mean_reward / n_trajs).item()

            if acc <= 1/2:
                perturb_params(model, -2*mu, seed)

            if (global_step+1) % logging_steps == 0:
                wandb.log({
                    "reward": mean_reward,
                    "epoch": epoch,
                    "accuracy": acc,
                }, step=global_step)
            
            print(f"Step: {global_step}, reward: {mean_reward}, accuracy: {acc}, time taken: {time_taken:.4f} seconds")

            n_w = 0
            n_trajs = 0
            mean_reward = 0.0
            global_step +=1
            start_time = time.time()
        
        else:
            perturb_params(model, -mu, seed)