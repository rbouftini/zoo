from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb, time

model_name = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained("checkpoints/SFTHIGH")
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

wandb.init(
    project="qwen3-dpo",
    name=f"qwen3-0.6b-dpo-{int(time.time())}",
)

training_args = DPOConfig(
    output_dir="checkpoints/DPO",
    report_to="wandb",         
    run_name=wandb.run.name,   
    logging_steps=50,          
    per_device_train_batch_size=12,
    num_train_epochs=1
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

trainer.train()

wandb.finish()