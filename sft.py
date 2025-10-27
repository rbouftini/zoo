from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 
import time 
import wandb 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Running on device:", device) 

n_epochs = 10
train_batch_size = 8
val_batch_size = 8
logging_steps = 50
eval_steps = 32 
max_lr = 1e-5 
wandb_save = False

model_name = "Qwen/Qwen3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
dataset = load_from_disk("../smoltalkIds").with_format(
                        "torch", columns=["input_ids", "prompt_len"]) 

wandb.init(
    project="sft-qwen3",
    name=f"qwen3-8B-sft-{int(time.time())}",
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
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    elif it < (n_steps - warmdown_steps): 
        return max_lr 
    else: 
        decay_ratio = (n_steps - it) / warmdown_steps 
        return max_lr * decay_ratio 
        
def get_val_loss(): 
    model.eval() 
    steps = eval_steps // val_batch_size 
    it = iter(loader_val) 
    total_loss = 0.0 
    with torch.inference_mode(): 
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
loader_train = DataLoader(dataset["train"].select(range(1000)), batch_size=train_batch_size, collate_fn=collate, shuffle=True, pin_memory=True) 

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto") 

for name, p in model.named_parameters():
    setattr(p, "_name", name)

model.config.use_cache = False
model.to(device) 

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False, maximize=False, adamw=True, fused=False):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, adamw=adamw)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']
            beta1, beta2 = group['betas']
            eps       = group['eps']
            wd        = group['weight_decay']
            amsgrad   = group['amsgrad']
            maximize  = group['maximize']
            use_adamw = group['adamw']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if maximize:
                    grad = -grad

                if (not use_adamw) and wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg     = state['exp_avg']
                exp_avg_sq  = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_raw = max_exp_avg_sq
                else:
                    denom_raw = exp_avg_sq

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t

                step_size = lr / bias_c1
                denom = (denom_raw / bias_c2).sqrt().add_(eps)

                update = exp_avg / denom

                if use_adamw and wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                p.add_(update, alpha=-step_size)

        return loss
    
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.0, weight_decay=0.0,
                 dampening=0.0, nesterov=False, maximize=False, fused=False):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        dampening=dampening, nesterov=nesterov, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']
            momentum  = group['momentum']
            weightdec = group['weight_decay']
            damp      = group['dampening']
            nesterov  = group['nesterov']
            maximize  = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.detach()
                if maximize:
                    d_p = -d_p

                if weightdec != 0:
                    d_p = d_p.add(p, alpha=weightdec)

                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - damp)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                update = d_p / (torch.abs(d_p)+1e-18)
                p.add_(update, alpha=-lr)

            return loss
    
optimizer = SGD(model.parameters())

n_steps = len(loader_train) * n_epochs
warmup_steps = int(n_steps * 0.1)
warmdown_steps = int(n_steps * 0.2)
train_loss = 0.0 
global_step = 0

print(f"Total number of steps: {n_steps}, warmup steps: {warmup_steps}, decay steps: {warmdown_steps}")

for epoch in range(n_epochs): 
    for step, x in enumerate(loader_train): 
        start_time = time.time() 

        lr = get_lr(global_step) 
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr 

        x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
        out = model(**x) 
        loss = out.loss 
        loss.backward() 

        optimizer.step() 
        optimizer.zero_grad() 
        
        train_loss += loss.item() / logging_steps 
        end_time = time.time() 
        time_taken = end_time - start_time 

        if (global_step+1) % logging_steps == 0: 
            val_loss = get_val_loss() 
            
            wandb.log({
                "train/loss_avg": train_loss,
                "train/lr": lr,
                "val/loss": val_loss,
                "time/step_s": time_taken,
                "epoch": epoch,
            }, step=global_step+1)

            print(f"Step:{global_step+1}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, lr: {lr:.4e}, time: {time_taken:.4f} seconds") 
            train_loss = 0.0 
        
        global_step += 1

tokenizer.eos_token = "<|im_end|>"

model.config.eos_token_id = tokenizer.eos_token
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

save_dir = "checkpoints/SFT"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

if wandb_save:
    artifact = wandb.Artifact("qwen3-8B-sft", type="model")
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)

wandb.finish()