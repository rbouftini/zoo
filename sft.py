from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 
import time 
import wandb 

device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Device:", device) 

n_epochs = 1 
train_batch_size = 8 
val_batch_size = 8
logging_steps = 50 
eval_steps = 32 
max_lr = 5e-5 
wandb_save = False

model_name = "Qwen/Qwen3-0.6B-Base" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
dataset = load_from_disk("../smoltalkIds").with_format(
                        "torch", columns=["input_ids", "prompt_len"]) 

wandb.init(
    project="sft-qwen3",
    name=f"qwen3-0.6b-sft-{int(time.time())}",
    config={
        "model_name": model_name,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "max_lr": max_lr,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "n_epochs": n_epochs,
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["test"]),
    },
)

def get_lr(it): 
    if it < (n_steps - warmdown_steps): 
        return max_lr 
    else: 
        decay_ratio = (n_steps - it) / warmdown_steps 
        return max_lr * decay_ratio 
        
def get_val_loss(): 
    model.eval() 
    steps = eval_steps // val_batch_size 
    it = iter(loader_val) 
    total_loss = 0.0 
    with torch.no_grad(): 
        for _ in range(steps): 
            x = next(it) 
            x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
            out = model(**x) 
            loss = out.loss 
            total_loss += loss.item() / steps 
            
    model.train()     
    return total_loss 
            
def collate(batch): 
    batch_ids = [x["input_ids"] for x in batch] 
    batch_ids = pad_sequence(batch_ids, batch_first=True, padding_value=tokenizer.pad_token_id) 
    attention_mask = (batch_ids != tokenizer.pad_token_id).long() 
    labels = batch_ids.clone() 
    labels[batch_ids == tokenizer.pad_token_id] = -100 
    for i, x in enumerate(batch): 
        labels[i, :x["prompt_len"]] = -100 
    return {"input_ids": batch_ids, "attention_mask": attention_mask, "labels": labels} 

loader_val = DataLoader(dataset["test"], batch_size=val_batch_size, collate_fn=collate, shuffle=True, pin_memory=True) 
loader_train = DataLoader(dataset["train"], batch_size=train_batch_size, collate_fn=collate, shuffle=True, pin_memory=True) 

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto") 
model.config.use_cache = False 
model.to(device) 
optimizer = torch.optim.AdamW(model.parameters(), fused=True) 

n_steps = len(loader_train) 
warmdown_steps = n_steps 
train_loss = 0.0 

for epoch in range(n_epochs): 
    for step, x in enumerate(loader_train): 
        start_time = time.time() 

        lr = get_lr(step) 
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr 

        x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
        out = model(**x) 
        loss = out.loss 
        loss.backward() 

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step() 
        optimizer.zero_grad() 
        
        train_loss += loss.item() / logging_steps 
        end_time = time.time() 
        time_taken = end_time - start_time 

        if (step+1) % logging_steps == 0: 
            val_loss = get_val_loss() 
            
            wandb.log({
                "train/loss_avg": train_loss,
                "train/grad_norm": float(grad_norm),
                "train/lr": lr,
                "val/loss": val_loss,
                "time/step_s": time_taken,
                "epoch": epoch,
            }, step=step)

            print(f"Step:{step+1}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, lr: {lr:.4e}, time: {time_taken:.4f} seconds") 
            train_loss = 0.0 

tokenizer.eos_token = "<|im_end|>"

model.config.eos_token_id = tokenizer.eos_token
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

save_dir = "checkpoints/SFT"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

if wandb_save:
    artifact = wandb.Artifact("qwen3-0.6b-sft", type="model")
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)

wandb.finish()