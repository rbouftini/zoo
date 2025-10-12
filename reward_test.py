import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
batch_size = 128

rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map=device,
    dtype = torch.bfloat16,
    num_labels=1,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def collate(batch):
    batch = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=False)
    batch = tokenizer(batch, return_tensors="pt", padding=True)
    return batch

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
n_examples = dataset.num_rows

chosen_loader = DataLoader(dataset["chosen"], batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)
rejected_loader = DataLoader(dataset["rejected"], batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

n_w = 0
for a_w, a_l in zip(chosen_loader, rejected_loader):
   a_w = {k: v.to(device, non_blocking=True) for k, v in a_w.items()} 
   a_l = {k: v.to(device, non_blocking=True) for k, v in a_l.items()} 

   time_start = time.time()
   with torch.no_grad():
    scores_w = rm(**a_w).logits
    scores_l = rm(**a_l).logits
    n_w += (scores_w > scores_l).long().sum(0).item()

   torch.cuda.synchronize()
   time_end = time.time()
   print(f"Time taken: {(time_end - time_start):.4f}")

acc = n_w/ n_examples
print(f"Reward model accuracy on Ultrafeedback: {acc:.4f}")