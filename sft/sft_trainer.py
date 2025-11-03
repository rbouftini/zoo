from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 
import time 
import wandb 
import os
from abc import ABC, abstractmethod
import argparse
import json
import math

# Temporarly
#os.environ["WANDB_MODE"] = "offline"

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--config")
args, _ = parent_parser.parse_known_args()

config = {}
if args.config:
    with open(args.config) as f:
        config = json.load(f)

parser = argparse.ArgumentParser(parents=[parent_parser])   
parser.add_argument("--model_name", help="Model", 
                    default="Qwen/Qwen3-8B", type=str)
parser.add_argument("--opt", help="Optimizer", 
                    choices=["Adam", "SignSGD", "RSS", "MSS", "SPSA", "R-AdaZO"], default="Adam")
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
                    default=0.05, type=float)
parser.add_argument("--save", help="Save model locally", action="store_true")

parser.set_defaults(**config)
args = parser.parse_args()

init_seed = 21
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

def log(train_loss, lr, val_loss, time_taken, epoch, global_step):
    wandb.log({
    "train/loss_avg": train_loss,
    "train/lr": lr,
    "val/loss": val_loss,
    "time/step_s": time_taken,
    "epoch": epoch,
    }, step=global_step+1)

    print(f"Step:{global_step+1}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, lr: {lr:.4e}, time: {time_taken:.4f} seconds") 

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
    name=f"{args.model_name}-{args.opt}",
    config=config,
    )
else:
    config["mu"] = args.mu
    wandb.init(
    project="sft",
    name=f"{args.model_name}-{args.opt}",
    config=config,
    )

if args.opt in ["Adam", "SignSGD"]:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto") 
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16) 

tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
model.config.use_cache = False
model.to(device) 

# Learning rate schedule (Warmup-stable-decay)
class WSD:
    def __init__(self, max_lr, n_steps, warmup_ratio=0.05, warmdown_ratio=0.05):
        self.max_lr = float(max_lr)
        self.n_steps = int(n_steps)
        self.warmup_steps = int(n_steps * warmup_ratio)
        self.warmdown_steps = int(n_steps * warmdown_ratio)

    def __call__(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        elif it < (self.n_steps - self.warmdown_steps):
            return self.max_lr
        else:
            decay_ratio = (self.n_steps - it) / self.warmdown_steps
            return self.max_lr * decay_ratio

# Learning rate schedule (linear-warmup-cosine-decay)
class CosineDecay:
    def __init__(self, max_lr, n_steps, warmup_ratio=0.05):
        self.max_lr = float(max_lr)
        self.n_steps = int(n_steps)
        self.warmup_steps = int(n_steps * warmup_ratio)
        self.warmdown_steps = self.n_steps - self.warmup_steps

    def __call__(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        else:
            decay_ratio = (it - self.warmup_steps) / (self.n_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return coeff * self.max_lr

n_steps = len(loader_train) * args.epochs
lr_sched = CosineDecay(args.lr, n_steps, args.warmup_ratio)
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
            for x in loader_train: 
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
                    log(train_loss, lr, val_loss, time_taken, epoch, global_step)
                    train_loss = 0.0 
                
                global_step += 1

class ZOTrainer(ABC):
    def __init__(self, model, lr_sched, init_seed):
        self.model = model
        self.lr_sched = lr_sched
        self.init_seed = init_seed

    def perturb_params(self, mu, seed, dist="normal", projected_grad=None):
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad: 
                    continue
                if dist == "rademacher":
                    r = 2 * torch.randint(0, 2, p.shape, generator=g, device=p.device, dtype=torch.float32) - 1
                else:
                    if dist != "normal":
                        print("Perturbation distribution is not valid (normal, rademacher). Reverting back to normal distribution.")

                    r = torch.randn(size=p.shape, device=device, generator=g, dtype=torch.float32)

                if projected_grad:
                    mu = -projected_grad * mu
                p.add_(r.to(p.dtype), alpha=mu)

    @abstractmethod
    def train(self):
        pass

class SPSATrainer(ZOTrainer):
    def __init__(self, model, lr_sched, mu_sched, init_seed):
        super().__init__(model, lr_sched, init_seed)
        self.mu_sched = mu_sched

    def train(self):
        train_loss = 0.0 
        global_step = 0
        for epoch in range(args.epochs): 
            for x in loader_train: 
                start_time = time.time() 
                seed = init_seed + global_step

                lr = self.lr_sched(global_step)
                mu  = self.mu_sched(global_step)

                losses = torch.empty(2)
                x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
                for i in range(2):
                    if i == 0:
                        self.perturb_params(mu, seed)
                    else:
                        self.perturb_params(-2*mu, seed)
                    
                    with torch.inference_mode():
                        out = self.model(**x) 
                        loss = out.loss
                        losses[i] = loss
                
                train_loss += losses[0].item() / args.logging_steps 
                end_time = time.time() 
                time_taken = end_time - start_time 

                self.perturb_params(mu, seed)

                projected_grad = (losses[0] - losses[1]) / (2*mu)
                self.perturb_params(lr, seed, projected_grad=projected_grad)

                if (global_step+1) % args.logging_steps == 0: 
                    val_loss = get_val_loss() 
                    log(train_loss, lr, val_loss, time_taken, epoch, global_step)
                    train_loss = 0.0 
                
                global_step += 1

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg     = state['exp_avg']
                exp_avg_sq  = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t

                step_size = lr / bias_c1
                denom = (exp_avg_sq/ bias_c2).sqrt().add_(eps)

                update = exp_avg / denom
                p.add_(update, alpha=-step_size)

class RAdaZO(ZOTrainer):
    def __init__(self, model, optimizer, lr_sched, mu_sched, init_seed):
        super().__init__(model, lr_sched, init_seed)
        self.mu_sched = mu_sched
        self.optimizer = optimizer
        
    def attach_grad(self, projected_grad, seed, dist="rademacher"):
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad: 
                    continue
                if dist == "rademacher":
                    r = 2 * torch.randint(0, 2, p.shape, generator=g, device=p.device, dtype=torch.float32) - 1
                else:
                    if dist != "normal":
                        print("Perturbation distribution is not valid (normal, rademacher). Reverting back to normal distribution.")

                    r = torch.randn(size=p.shape, device=device, generator=g, dtype=torch.float32)

                grad = projected_grad * r
                p.grad = grad
        
    def train(self):
        train_loss = 0.0 
        global_step = 0
        for epoch in range(args.epochs): 
            for x in loader_train: 
                start_time = time.time() 
                seed = init_seed + global_step

                lr = self.lr_sched(global_step)
                for param_group in self.optimizer.param_groups: 
                    param_group['lr'] = lr 
                
                mu  = self.mu_sched(global_step)

                losses = torch.empty(2)
                x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
                for i in range(2):
                    if i == 0:
                        self.perturb_params(mu, seed)
                    else:
                        self.perturb_params(-2*mu, seed)
                    
                    with torch.inference_mode():
                        out = self.model(**x) 
                        loss = out.loss
                        losses[i] = loss
                
                train_loss += losses[0].item() / args.logging_steps 
                end_time = time.time() 
                time_taken = end_time - start_time 

                self.perturb_params(mu, seed)

                projected_grad = (losses[0] - losses[1]) / (2*mu)
                self.attach_grad(projected_grad, seed)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if (global_step+1) % args.logging_steps == 0: 
                    val_loss = get_val_loss() 
                    log(train_loss, lr, val_loss, time_taken, epoch, global_step)
                    train_loss = 0.0 
                
                global_step += 1
    
class RSSTrainer(ZOTrainer):
    def __init__(self, model, lr_sched, init_seed):
        super().__init__(model, lr_sched, init_seed)
    
    def train(self):
        train_loss = 0.0 
        global_step = 0
        for epoch in range(args.epochs): 
            for x in loader_train: 
                start_time = time.time() 
                seed = init_seed + global_step

                lr = self.lr_sched(global_step) 
                losses = torch.empty(2)
                x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
                for i in range(2):
                    if i == 0:
                        self.perturb_params(lr, seed)
                    else:
                        self.perturb_params(-2*lr, seed)
                    
                    with torch.inference_mode():
                        out = self.model(**x) 
                        loss = out.loss
                        losses[i] = loss
                
                train_loss += losses[0].item() / args.logging_steps 
                end_time = time.time() 
                time_taken = end_time - start_time 

                self.perturb_params(lr, seed)

                if losses[0] < losses[1]:
                    self.perturb_params(lr, seed)
                elif losses[0] > losses[1]:
                    self.perturb_params(-lr, seed)

                if (global_step+1) % args.logging_steps == 0: 
                    val_loss = get_val_loss() 
                    log(train_loss, lr, val_loss, time_taken, epoch, global_step)
                    train_loss = 0.0 
                
                global_step += 1

class MSSTrainer(ZOTrainer):
        def __init__(self, model, lr_sched, init_seed):
            super().__init__(model, lr_sched, init_seed)
    
        def train(self):
            train_loss = 0.0 
            global_step = 0
            for epoch in range(args.epochs): 
                for x in loader_train: 
                    start_time = time.time() 
                    seed = init_seed + global_step

                    lr = lr_sched(global_step) 
                    losses = torch.empty(2)
                    x = {k: v.to(device, non_blocking=True) for k, v in x.items()} 
                    for i in range(2):
                        if i == 1:
                            self.perturb_params(lr, seed)
                        
                        with torch.inference_mode():
                            out = model(**x) 
                            loss = out.loss
                            losses[i] = loss
                    
                    train_loss += losses[0].item() / args.logging_steps 
                    end_time = time.time() 
                    time_taken = end_time - start_time 
                    
                    self.perturb_params(-lr, seed)

                    if losses[1] < losses[0]:
                        self.perturb_params(lr, seed)

                    if (global_step+1) % args.logging_steps == 0: 
                        val_loss = get_val_loss() 
                        log(train_loss, lr, val_loss, time_taken, epoch, global_step)
                        train_loss = 0.0 
                    
                    global_step += 1

match args.opt:
    case "Adam":
        optimizer = torch.optim.Adam(model.parameters(), fused=True)
        trainer = BpTrainer(model, optimizer, lr_sched)
    case "SignSGD":
        optimizer = SignSGD(model.parameters(), fused=True)
        trainer = BpTrainer(model, optimizer, lr_sched)
    case "RSS":
        trainer = RSSTrainer(model, lr_sched, init_seed)
    case "MSS":
        trainer = MSSTrainer(model, lr_sched, init_seed)
    case "SPSA":
        mu_sched = WSD(max_lr=args.mu, n_steps=n_steps, warmdown_ratio=0, warmup_ratio=0)
        trainer = SPSATrainer(model, lr_sched, mu_sched, init_seed)
    case "R-AdaZO":
        mu_sched = WSD(max_lr=args.mu, n_steps=n_steps, warmdown_ratio=0, warmup_ratio=0)
        optimizer = Adam(model.parameters())
        trainer = RAdaZO(model, optimizer, lr_sched, mu_sched, init_seed)

trainer.train()

if args.save:
    tokenizer.eos_token = "<|im_end|>"
    model.config.eos_token_id = tokenizer.eos_token
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    save_dir = "checkpoints/SFT"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

wandb.finish()