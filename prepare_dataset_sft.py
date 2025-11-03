from datasets import load_dataset
from transformers import AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = "HuggingFaceTB/smol-smoltalk"
N_TRAIN_EX = 100000
N_TEST_EX = 5000

dataset = load_dataset(dataset_name)
dataset = dataset.remove_columns("source")
dataset = dataset.shuffle(seed=21)

dataset["train"] = dataset["train"].select(range(min(N_TRAIN_EX, dataset["train"].num_rows)))
dataset["test"] = dataset["test"].select(range(min(N_TEST_EX, dataset["test"].num_rows)))

def format(row):
    messages = row["messages"]
    # Messages is a list of dictionaries for the conversation
    # We will return the tokenized conversation and mask for the tokens to account for in the SFT loss.
    # Mask==0 for user or system content, else its 1
    input_ids, mask = [], []
    def add_tokens(ids, mask_val):
        if isinstance(ids, int):
            ids = [ids]
        input_ids.extend(ids)
        mask.extend([mask_val] * len(ids))

    # Sometimes the conversation starts with a system prompt
    if messages[0]["role"] == "system":
        msg = "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
        ids = tokenizer.encode(msg)
        add_tokens(ids, 0)
        messages = messages[1:]
    
    for message in messages:
        if message["role"] == "user":
            msg = "<|im_start|>user\n" + message["content"] + "<|im_end|>\n"
            ids = tokenizer.encode(msg)
            add_tokens(ids, 0)
        else:
            assert message["role"] == "assistant"
            msg =  "<|im_start|>assistant\n<think>\n\n</think>\n\n" + message["content"] + "<|im_end|>\n"
            ids = tokenizer.encode(msg)
            add_tokens(ids, 1)

    return {
        "input_ids": input_ids,
        "mask": mask
    }

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = dataset.map(format, remove_columns=dataset["train"].column_names, num_proc=16)

dataset = dataset.filter(lambda ex: len(ex["input_ids"])<=1024, num_proc=16)
print(dataset)

dataset.save_to_disk("../smoltalkIds")