from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = "HuggingFaceTB/smol-smoltalk"
flag = "<think>\n\n</think>\n"
N_TRAIN_EX = 800
N_TEST_EX = 50

dataset = load_dataset(dataset_name)
dataset = dataset.remove_columns("source")
dataset = dataset.shuffle(seed=7)

dataset["train"] = dataset["train"].select(range(min(N_TRAIN_EX, dataset["train"].num_rows)))
dataset["test"] = dataset["test"].select(range(min(N_TEST_EX, dataset["test"].num_rows)))

def format(row):
    # For now, we format our dataset only as prompt and completion (not conversational)
    if row["messages"][0]["role"] == "system":
        row["messages"] = row["messages"][:3]
    else:
        row["messages"] = row["messages"][:2]

    full_text = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, enable_thinking=False, tokenize=False)
    prompt_len = len(tokenizer(full_text[:full_text.find(flag)])["input_ids"])
    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    input_ids = input_ids.tolist()[0]

    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len
    }

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = dataset.map(format, remove_columns=dataset["train"].column_names)