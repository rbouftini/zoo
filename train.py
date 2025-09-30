from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
dataset_name = "HuggingFaceTB/smol-smoltalk"
N_TRAIN_EX = 5
N_TEST_EX = 2

dataset = load_dataset(dataset_name)
dataset = dataset.remove_columns("source")
dataset = dataset.shuffle(seed=42)

dataset["train"] = dataset["train"].select(range(min(N_TRAIN_EX, dataset["train"].num_rows)))
dataset["test"] = dataset["test"].select(range(min(N_TEST_EX, dataset["test"].num_rows)))

def format(row):
    if row["messages"][0]["role"] == "system":
        row["messages"] = row["messages"][:3]
    else:
        row["messages"] = row["messages"][:2]
    return row
    
dataset = dataset.map(format)
tokenizer = AutoTokenizer.from_pretrained(model_name)
msg = [{"role":"user", "content": "Hello, how are you?"}]

# enable_thinking = False will append <think> </think> tokens (only if add_generation_prompt=True) so the model doesn't predict thinking tokens
print(tokenizer.apply_chat_template(dataset["test"]["messages"][0], tokenize=False, add_generation_prompt=False, enable_thinking=False))