from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from transformers import DefaultDataCollator
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader 
import time 
import wandb 
import os
from abc import ABC, abstractmethod
import argparse
import json
import math
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Temporarly
os.environ["WANDB_MODE"] = "online"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

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
                    choices=["Adafactor", "RSS", "MSS", "SPSA", "R-AdaZO"], default="Adafactor")
parser.add_argument("--epochs", help="Number of epochs",
                    default=2, type=int)
parser.add_argument("--train_batch_size", help="Batch size of training data",
                    default=1, type=int)
parser.add_argument("--grad_accum_steps", help="Gradient accumulation steps",
                    default=1, type=int)
parser.add_argument("--val_batch_size", help="Batch size of validation data",
                    default=1, type=int)
parser.add_argument("--logging_steps", help="Number of update steps between two logs",
                    default=1, type=int)
parser.add_argument("--eval_steps", help="Number of evaluation steps",
                    default=1, type=int)
parser.add_argument("--saving_steps", help="Number of steps before checkpointing",
                    default=100, type=int)
parser.add_argument("--lr", help="Maximum learning rate",
                    default=3e-4, type=float)
parser.add_argument("--mu", help="Perturbation parameter",
                    default=3e-4, type=float)
parser.add_argument("--warmup_ratio", help="Ratio of total number of steps for a linear warmup",
                    default=0.03, type=float)
parser.add_argument("--warmdown_ratio", help="Ratio of total number of steps for a linear warmdown",
                    default=0.05, type=float)
parser.add_argument("--save", help="Save model locally", action="store_true")
parser.add_argument("--resume_training", help="Resume training from saved checkpoint", action="store_true")

parser.set_defaults(**config)
args = parser.parse_args()

init_seed = 16
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Running on device:", device) 

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = ddp_rank == 0
else:
    master_process = True

# Helper functions
def gen_batch(loader):
    while True:
        for batch in loader:
            yield batch

def get_val_loss():
    model.eval() 
    steps = args.eval_steps // args.val_batch_size
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode(): 
        for _ in range(steps): 
            x = next(it_val)
            out = model(**x) 
            batch_loss_mean = out.loss

            if "labels" in x:
                token_count = (x["labels"] != -100).sum()
                batch_loss_sum = batch_loss_mean.float() * token_count
            
            total_loss += batch_loss_sum
            total_tokens += token_count

    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    val_loss = total_loss / total_tokens
    
    model.train()     
    return val_loss.item()
            
def collate(batch): 
    batch_ids, masks = [], []
    for row in batch:
        batch_ids.append(row["input_ids"])
        masks.append(row["mask"])

    batch_ids = pad_sequence(batch_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0) 
    attention_mask = (batch_ids != tokenizer.pad_token_id).long() 
    labels = batch_ids.clone() 
    labels[masks == 0] = -100 

    return {"input_ids": batch_ids, "attention_mask": attention_mask, "labels": labels} 

@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    return_flash_attn_kwargs: bool = True
    return_position_ids: bool = True
    return_seq_idx: bool = True
    separator_id: int = -100

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        if self.return_position_ids:
            pos_ids = []
        if self.return_seq_idx:
            seq_idx = []

        is_labels_provided = "labels" in features[0]
        has_mask = "mask" in features[0]

        ret = {"input_ids": [], "labels": []}
        separator = torch.tensor(
            [separator_id],
            dtype=features[0]["input_ids"].dtype,
            device=features[0]["input_ids"].device,
        )

        for s_idx, item in enumerate(features):
            input_ids = item["input_ids"]
            ret["input_ids"].append(input_ids)

            if is_labels_provided:
                labels = item["labels"]
            elif has_mask:
                labels = input_ids.clone()
                labels[item["mask"] == 0] = -100
            else:
                labels = input_ids

            ret["labels"].append(separator)
            ret["labels"].append(labels[1:])

            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))
            if self.return_position_ids:
                pos_ids.append(torch.arange(input_ids.numel(), device=input_ids.device))
            if self.return_seq_idx:
                seq_idx.append(torch.full_like(input_ids, s_idx, dtype=torch.int32))

        base_device = features[0]["input_ids"].device 

        if self.return_flash_attn_kwargs:
            ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = torch.tensor(
                cu_seq_lens, dtype=torch.int32, device=base_device
            )
            ret["max_length_q"] = ret["max_length_k"] = max_length
        if self.return_position_ids:
            ret["position_ids"] = torch.cat(pos_ids, dim=0)[None]
        if self.return_seq_idx:
            ret["seq_idx"] = torch.cat(seq_idx, dim=0)[None]

        ret["input_ids"] = torch.cat(ret["input_ids"], dim=0)[None]
        ret["labels"] = torch.cat(ret["labels"], dim=0)[None]

        ret["attention_mask"] = torch.ones_like(ret["input_ids"], dtype=torch.int32)

        for k, v in list(ret.items()):
            if isinstance(v, torch.Tensor):
                ret[k] = v.to(device)

        return ret

def log(train_loss, lr, val_loss, time_taken, global_step):
    wandb.log({
    "train/loss_avg": train_loss,
    "train/lr": lr,
    "val/loss": val_loss,
    "time/step_s": time_taken,
    }, step=global_step)

    print(f"Step:{global_step}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}, lr: {lr:.4e}, time: {time_taken:.4f} seconds") 

# Load Dataset (Default for now is Smoltalk)
dataset = load_from_disk("../../smoltalkIds").with_format(
                        "torch", columns=["input_ids", "mask"])

if master_process:
    print(f"Number of training examples: {dataset.num_rows}")
    print("Gradient accumulation steps:", args.grad_accum_steps)

if ddp:
    sampler_val = DistributedSampler(dataset["test"], shuffle=False)
    sampler_train = DistributedSampler(dataset["train"], shuffle=False)
else:
    sampler_val, sampler_train = None, None

collate_fn = TensorDataCollatorWithFlattening()
loader_val = DataLoader(dataset["test"], batch_size=args.val_batch_size, 
                        collate_fn=collate_fn, shuffle=False,sampler=sampler_val) 
loader_train = DataLoader(dataset["train"], batch_size=args.train_batch_size, 
                        collate_fn=collate_fn, shuffle=False, sampler=sampler_train) 
it_val = gen_batch(loader_val)
it_train = gen_batch(loader_train)

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

if args.opt == "Adafactor":
    if master_process:
        wandb.init(
        project="sft",
        name=f"{args.model_name}-{args.opt}",
        config=config,
        )
else:
    config["mu"] = args.mu
    if master_process:
        wandb.init(
        project="sft",
        name=f"{args.model_name}-{args.opt}",
        config=config,
        )

model_dir = f"checkpoints/{args.model_name}" if args.resume_training else args.model_name

if args.opt == "Adafactor":
    raw_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto", attn_implementation="flash_attention_2") 
else:
    raw_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, attn_implementation="flash_attention_2")
    raw_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_dir) 
raw_model.config.use_cache = False
raw_model.to(device) 

model = DDP(raw_model, device_ids=[ddp_local_rank]) if ddp else raw_model

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
        
n_steps = len(loader_train) * args.epochs // args.grad_accum_steps
lr_sched = WSD(args.lr, n_steps, args.warmup_ratio, args.warmdown_ratio)
if master_process:
    print(f"Total number of steps: {n_steps}, warmup steps: {lr_sched.warmup_steps}, decay steps: {lr_sched.warmdown_steps}")

class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, fused=False):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.detach()
                update = d_p / (torch.abs(d_p)+1e-9)
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
        
        if args.resume_training:
            if master_process:
                print("Resuming Training")

            checkpoint_dir = os.path.join(model_dir, "state.pt")
            state_dict = torch.load(checkpoint_dir, map_location="cpu")
            self.optimizer.load_state_dict(state_dict["optimizer_state"])
            global_step = state_dict["step"]
            skipping_steps = global_step * args.grad_accum_steps
            for _ in range(skipping_steps):
                next(it_train)

        while True: 
            torch.cuda.synchronize()
            start_time = time.time() 

            lr = self.lr_sched(global_step) 
            for param_group in self.optimizer.param_groups: 
                param_group['lr'] = lr 

            for _ in range(args.grad_accum_steps):
                x = next(it_train)
                out = self.model(**x) 
                loss = out.loss 
                loss = loss / args.grad_accum_steps
                train_loss += loss.detach().item()
                loss.backward() 

            self.optimizer.step() 
            self.optimizer.zero_grad(set_to_none=True) 
            
            torch.cuda.synchronize()
            end_time = time.time() 
            time_taken = end_time - start_time 

            if global_step % args.logging_steps == 0: 
                if global_step != 0:
                    train_loss = train_loss / args.logging_steps
                val_loss = get_val_loss() 
                if master_process:
                    log(train_loss, lr, val_loss, time_taken, global_step)
                train_loss = 0.0
            
            if master_process and args.save and (global_step+1) % args.saving_steps == 0:
                save_dir = f"checkpoints/{args.model_name}-{args.opt}-Step-{global_step}"
                raw_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                state = {
                    "optimizer_state": self.optimizer.state_dict(),
                    "step": global_step + 1, 
                }
                torch.save(state, os.path.join(save_dir, f"state.pt"))

            global_step += 1

            if global_step >= n_steps:
                break
                
class ZOTrainer(ABC):
    def __init__(self, model, lr_sched, init_seed):
        self.model = model
        self.lr_sched = lr_sched
        self.init_seed = init_seed

    def perturb_params(self, mu, seed, dist="normal", projected_grad=None):
        g = torch.Generator(device=device).manual_seed(seed)
        scale = mu
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad: 
                    continue
                if dist == "rademacher":
                    r = 2 * torch.randint(0, 2, p.shape, generator=g, device=p.device, dtype=p.dtype) - 1
                else:
                    if dist != "normal":
                        print("Perturbation distribution is not valid (normal, rademacher). Reverting back to normal distribution.")

                    r = torch.randn(size=p.shape, device=device, generator=g, dtype=p.dtype)

                if projected_grad:
                    scale = -projected_grad * mu
                p.add_(r, alpha=scale)
            
            torch.cuda.synchronize()

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

        if args.resume_training:
            if master_process:
                print("Resuming Training")

            checkpoint_dir = os.path.join(model_dir, "state.pt")
            state_dict = torch.load(checkpoint_dir, map_location="cpu")
            global_step = state_dict["step"]
            skipping_steps = global_step * args.grad_accum_steps
            for _ in range(skipping_steps):
                next(it_train)
        
        while True:
            torch.cuda.synchronize()
            start_time = time.time() 
            seed = init_seed + global_step

            lr = self.lr_sched(global_step)
            mu  = self.mu_sched(global_step)

            x = next(it_train)
            losses = torch.empty(2, device=device)
            for i in range(2):
                if i == 0:
                    self.perturb_params(mu, seed)
                else:
                    self.perturb_params(-2*mu, seed)
                
                with torch.inference_mode():
                    out = self.model(**x) 
                    loss = out.loss
                    losses[i] = loss
            
            train_loss += losses[0].item()

            self.perturb_params(mu, seed)
            if ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)

            projected_grad = (losses[0] - losses[1]) / (2*mu)
            self.perturb_params(lr, seed, projected_grad=projected_grad)

            torch.cuda.synchronize()
            end_time = time.time() 
            time_taken = end_time - start_time 

            if global_step % args.logging_steps == 0: 
                if global_step != 0:
                    train_loss = train_loss / args.logging_steps
                val_loss = get_val_loss() 
                if master_process:
                    log(train_loss, lr, val_loss, time_taken, global_step)
                train_loss = 0.0 

            if master_process and args.save and (global_step+1) % args.saving_steps == 0:
                save_dir = f"checkpoints/{args.model_name}-{args.opt}-Step-{global_step}"
                raw_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)         
                state = {
                    "step": global_step + 1, 
                }
                torch.save(state, os.path.join(save_dir, f"state.pt"))
                
            global_step += 1

            if global_step >= n_steps:
                break

class ZeroAdam(torch.optim.Optimizer):
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
        
    def attach_grad(self, projected_grad, seed, dist="normal"):
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad: 
                    continue
                if dist == "rademacher":
                    r = 2 * torch.randint(0, 2, p.shape, generator=g, device=p.device, dtype=p.dtype) - 1
                else:
                    if dist != "normal":
                        print("Perturbation distribution is not valid (normal, rademacher). Reverting back to normal distribution.")

                    r = torch.randn(size=p.shape, device=device, generator=g, dtype=p.dtype)

                grad = projected_grad * r
                p.grad = grad
        
    def train(self):
        train_loss = 0.0 
        global_step = 0
        
        if args.resume_training:
            if master_process:
                print("Resuming Training")

            checkpoint_dir = os.path.join(model_dir, "state.pt")
            state_dict = torch.load(checkpoint_dir, map_location="cpu")
            self.optimizer.load_state_dict(state_dict["optimizer_state"])
            global_step = state_dict["step"]
            skipping_steps = global_step * args.grad_accum_steps
            for _ in range(skipping_steps):
                next(it_train)

        while True:
            torch.cuda.synchronize()
            start_time = time.time() 
            seed = init_seed + global_step

            lr = self.lr_sched(global_step)
            for param_group in self.optimizer.param_groups: 
                param_group['lr'] = lr 
            
            mu  = self.mu_sched(global_step)

            losses = torch.empty(2, device=device)
            x = next(it_train)
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

            self.perturb_params(mu, seed)

            if ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
                
            projected_grad = (losses[0] - losses[1]) / (2*mu)
            self.attach_grad(projected_grad, seed)

            self.optimizer.step()
            self.optimizer.zero_grad()

            torch.cuda.synchronize()
            end_time = time.time() 
            time_taken = end_time - start_time

            if (global_step+1) % args.logging_steps == 0: 
                val_loss = get_val_loss() 
                if master_process:
                    log(train_loss, lr, val_loss, time_taken, global_step)
                train_loss = 0.0 

            if master_process and args.save and (global_step+1) % args.saving_steps == 0:
                save_dir = f"checkpoints/{args.model_name}-{args.opt}-Step-{global_step}"
                raw_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)         
                state = {
                    "optimizer_state": self.optimizer.state_dict(),
                    "step": global_step + 1, 
                }
                torch.save(state, os.path.join(save_dir, f"state.pt"))
            
            global_step += 1

            if global_step >= n_steps:
                break
    
class RSSTrainer(ZOTrainer):
    def __init__(self, model, lr_sched, init_seed):
        super().__init__(model, lr_sched, init_seed)
    
    def train(self):
        train_loss = 0.0 
        global_step = 0

        if args.resume_training:
            if master_process:
                print("Resuming Training")

            checkpoint_dir = os.path.join(model_dir, "state.pt")
            state_dict = torch.load(checkpoint_dir, map_location="cpu")
            global_step = state_dict["step"]
            skipping_steps = global_step * args.grad_accum_steps
            for _ in range(skipping_steps):
                next(it_train)

        while True:
            torch.cuda.synchronize()
            start_time = time.time() 
            seed = init_seed + global_step

            lr = self.lr_sched(global_step) 
            losses = torch.empty(2, device=device)
            x = next(it_train)

            for i in range(2):
                if i == 0:
                    self.perturb_params(lr, seed)
                else:
                    self.perturb_params(-2*lr, seed)
                
                with torch.inference_mode():
                    out = self.model(**x) 
                    loss = out.loss
                    losses[i] = loss
            
            train_loss += losses[0].item()

            self.perturb_params(lr, seed)

            if ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)

            if losses[0] < losses[1]:
                self.perturb_params(lr, seed)
            elif losses[0] > losses[1]:
                self.perturb_params(-lr, seed)

            torch.cuda.synchronize()
            end_time = time.time() 
            time_taken = end_time - start_time

            if global_step % args.logging_steps == 0: 
                if global_step != 0:
                    train_loss = train_loss / args.logging_steps
                val_loss = get_val_loss() 
                if master_process:
                    log(train_loss, lr, val_loss, time_taken, global_step)
                train_loss = 0.0 
            
            if master_process and args.save and (global_step+1) % args.saving_steps == 0:
                save_dir = f"checkpoints/{args.model_name}-{args.opt}-Step-{global_step}"
                raw_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)         
                state = {
                    "step": global_step + 1, 
                }
                torch.save(state, os.path.join(save_dir, f"state.pt"))
        
            global_step += 1

            if global_step >= n_steps:
                break
    
class MSSTrainer(ZOTrainer):
        def __init__(self, model, lr_sched, init_seed):
            super().__init__(model, lr_sched, init_seed)
            self.margin = 1e4
    
        def train(self):
            train_loss = 0.0 
            global_step = 0
            seed = init_seed
            lr = args.lr

            if args.resume_training:
                if master_process:
                    print("Resuming Training")

                checkpoint_dir = os.path.join(model_dir, "state.pt")
                state_dict = torch.load(checkpoint_dir, map_location="cpu")
                global_step = state_dict["step"]
                skipping_steps = global_step * args.grad_accum_steps
                for _ in range(skipping_steps):
                    next(it_train)

            while True:
                start_time = time.time() 
                losses = torch.empty(2, device=device)
                x = next(it_train)

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

                if ddp:
                    dist.all_reduce(losses, op=dist.ReduceOp.AVG)

                if losses[1] < losses[0] - self.margin*(lr**2):
                    self.perturb_params(lr, seed)
                else:
                    seed = init_seed + global_step
                    lr = args.lr / np.sqrt(global_step+1)
                    
                if (global_step+1) % args.logging_steps == 0: 
                    val_loss = get_val_loss() 
                    if master_process:
                        log(train_loss, lr, val_loss, time_taken, global_step)
                    train_loss = 0.0 
                
                if master_process and args.save and (global_step+1) % args.saving_steps == 0:
                    save_dir = f"checkpoints/{args.model_name}-{args.opt}-Step-{global_step}"
                    raw_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)         
                    state = {
                        "step": global_step + 1, 
                    }
                    torch.save(state, os.path.join(save_dir, f"state.pt"))
        
                global_step += 1

                if global_step >= n_steps:
                    break

match args.opt:
    case "Adafactor":
        optimizer = torch.optim.Adam(model.parameters())
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
        optimizer = ZeroAdam(model.parameters())
        trainer = RAdaZO(model, optimizer, lr_sched, mu_sched, init_seed)

trainer.train()

if args.save and master_process:
    save_dir = f"checkpoints/{args.model_name}-SFT-Final"
    raw_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if ddp:
    dist.destroy_process_group()

wandb.finish()