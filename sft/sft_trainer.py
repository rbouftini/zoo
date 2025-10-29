from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 
import time 
import wandb 
import os
import math
import argparse

# Temporarly
os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model", required=True, type=str)
parser.add_argument("--opt", help="Optimizer", 
                    choices=["Adam, SignSGD, RSS, MSS, SPSA, R-AdaZO"], default="Adam")
parser.add_argument("--epochs", help="Number of epochs",
                    default=2, type=int)
parser.add_argument("--train_batch_size", help="Batch size of training data",
                    default=1, type=int)
parser.add_argument("--val_batch_size", help="Batch size of validation data",
                    default=1, type=int)
parser.add_argument("--logging_steps", help="Number of update steps between two logs",
                    default=1, type=int)
parser.add_argument("--eval_steps", help="Number of evaluation steps",
                    default=1, type=int)
parser.add_argument("--lr", help="Maximum learning rate",
                    default=3e-4, type=float)
parser.add_argument("--mu", help="Perturbation parameter",
                    default=3e-4, type=float)
parser.add_argument("--warmup_ratio", help="Ratio of total number of steps for a linear warmup",
                    default=0.03, type=float)
parser.add_argument("--warmdown_ratio", help="Ratio of total number of steps for a linear warmdown",
                    default=0.2, type=float)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Running on device:", device) 

# Helper functions
def get_val_loss(): 
    model.eval() 
    steps = args.eval_steps // args.val_batch_size 
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

# Load Dataset (Default for now is Smoltalk)
dataset = load_from_disk("../../smoltalkIds").with_format(
                        "torch", columns=["input_ids", "prompt_len"])
print(f"Number of training examples: {dataset.num_rows}")

loader_val = DataLoader(dataset["test"], batch_size=args.val_batch_size, 
                        collate_fn=collate, shuffle=True, pin_memory=True) 
loader_train = DataLoader(dataset["train"], batch_size=args.train_batch_size, 
                        collate_fn=collate, shuffle=True, pin_memory=True) 

# Wandb Setup
config={
    "train_batch_size": args.train_batch_size,
    "val_batch_size": args.val_batch_size,
    "max_lr": args.lr,
    "logging_steps": args.logging_steps,
    "eval_steps": args.eval_steps,
    "n_epochs": args.epochs,
    "train_examples": len(dataset["train"]),
    "val_examples": len(dataset["test"]),
}

if args.opt in ["Adam", "SignSGD"]:
    wandb.init(
    project="sft",
    name=f"{args.model_name}-{args.opt}-{int(time.time())}",
    config=config,
    )
else:
    config["mu"] = args.mu
    wandb.init(
    project="sft",
    name=f"{args.model_name}-{args.opt}-{int(time.time())}",
    config=config,
    )

if args.opt in ["Adam", "SignSGD"]:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto") 
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32) 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
model.config.use_cache = False
model.to(device) 

# Learning rate schedule (Warmup-stable-decay)
class WSD:
    def __init__(self, max_lr, n_steps, warmup_ratio=0.1, warmdown_ratio=0.2):
        self.max_lr = float(max_lr)
        self.n_steps = int(n_steps)
        self.warmup_steps = int(n_steps * warmup_ratio)
        self.warmdown_steps = int(n_steps * warmdown_ratio)

    def __call__(self, it: int) -> float:
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        elif it < (self.n_steps - self.warmdown_steps):
            return self.max_lr
        else:
            decay_ratio = (self.n_steps - it) / self.warmdown_steps
            return self.max_lr * decay_ratio

n_steps = len(loader_train) * args.epochs
lr_sched = WSD(args.lr, n_steps, args.warmup_ratio, args.warmdown_ratio)
print(f"Total number of steps: {n_steps}, warmup steps: {lr_sched.warmup_steps}, decay steps: {lr_sched.warmdown_steps}")

class SignSGD(torch.optim.Optimizer):
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
        
class BpTrainer():
    def __init__(self, model, optimizer, lr_sched):
        self.model = model
        self.lr_sched = lr_sched
        self.optimizer = optimizer

    def train(self):
        train_loss = 0.0 
        global_step = 0
        for epoch in range(args.epochs): 
            for step, x in enumerate(loader_train): 
                start_time = time.time() 

                lr = self.lr_sched(global_step) 
                for param_group in self.optimizer.param_groups: 
                    param_group['lr'] = lr 

                x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
                out = self.model(**x) 
                loss = out.loss 
                loss.backward() 

                self.optimizer.step() 
                self.optimizer.zero_grad() 
                
                train_loss += loss.item() / args.logging_steps 
                end_time = time.time() 
                time_taken = end_time - start_time 

                if (global_step+1) % args.logging_steps == 0: 
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

if args.opt == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), fused=True)
    trainer = BpTrainer(model, optimizer, lr_sched)
elif args.opt == "SignSGD":
    optimizer = SignSGD(model.parameters(), fused=True)
    trainer = BpTrainer(model, optimizer, lr_sched)

trainer.train()