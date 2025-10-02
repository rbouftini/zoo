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

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")

# enable_thinking = False will append <think> </think> tokens (only if add_generation_prompt=True) so the model doesn't predict thinking tokens
#print(tokenizer.apply_chat_template(dataset["test"]["messages"][0], tokenize=False, add_generation_prompt=False, enable_thinking=False))

class SFTDataCollator:
    def __call__(self, batch):
        input_ids_list = [torch.tensor(b["input_ids"]) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        for i, b in enumerate(batch):
            prompt_len = b["prompt_len"] 
            labels[i, :prompt_len] = -100 
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

data_collator = SFTDataCollator()

training_args = TrainingArguments(
    output_dir="qwen_sft_demo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,    
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,   
    eval_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=20,
    logging_steps=20,
    save_steps=0,                    
    report_to=[],                     
    #bf16=True,                     
    disable_tqdm=False,             
    remove_unused_columns=False, 
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    args=training_args,
    eval_dataset=dataset["test"]
)

trainer.train()

msg = [{"role":"user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False, return_tensors="pt").to(device)

output_ids = model.generate(input_ids, max_new_tokens=50)
answer = tokenizer.decode(output_ids[0])
print(answer)