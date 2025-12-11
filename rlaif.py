from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import asyncio
from openai import AsyncOpenAI

# Parameters
model = "Qwen/Qwen3-1.7B"
n_prompts = 2
n_responses_per_prompt = 2
device = "cuda"
init_seed = 16
mu = 1e-3
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
prompt_template = """Consider the following instruction and two responses. Which response would be \
preferred by someone who deeply loves humanity and has humanity's best interests \
at heart? Respond with the answer you prefer within \\boxed{{}} (either A or B)

Instruction:
{prompt}

Response A:
{response_a}

Response B:
{response_b}
"""

# Helper functions
def collate_fn(batch):
    messages = [[{"role": "user", "content": x}] for x in batch]
    formatted_prompts = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    
    return batch, formatted_prompts

def get_prompts(batch, raw_prompts):
    # outputs is a list of two elements
    # each element is a list containing RequestOuput objects for each prompt
    # .outputs to access the responses
    # the responses will be a list with the number of completions to access with .text
    assert len(batch) == 2 
    prompts_labeler = []
    
    for i in range(len(raw_prompts)):
        for j in range(n_responses_per_prompt):
            formatted_prompt = prompt_template.format(
                prompt=raw_prompts[i], 
                response_a=batch[0][i].outputs[j].text,
                response_b=batch[1][i].outputs[j].text
            )
            prompts_labeler.append(formatted_prompt)
    
    return prompts_labeler

async def request(prompt):
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0, # Greedy Decoding
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content

async def get_responses(prompts):
    tasks = [request(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Create client / AI Labeler
client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# Configure the sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=1, n=n_responses_per_prompt, max_tokens=8192)

# Initialize the vLLM engine with worker extension
vllm_model = LLM(model=model, worker_extension_cls="utils.myworker.MyWorker")

# Prepare the dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:4000]")
train_loader = DataLoader(dataset=dataset["prompt"], batch_size=n_prompts, collate_fn=collate_fn, shuffle=False, pin_memory=True)

global_step = 0
for raw_prompts, formatted_prompts in train_loader:
    seed = init_seed + global_step
    outputs = []

    # Generate responses
    for i in range(2):
        if i == 0:
            vllm_model.collective_rpc("perturb_params", args=(mu, seed))
        else:
            vllm_model.collective_rpc("perturb_params", args=(-2*mu, seed))

        outputs.append(vllm_model.generate(formatted_prompts, sampling_params))

    # Restore current parameters
    vllm_model.collective_rpc("perturb_params", args=(mu, seed))

    # Prepare prompts for AI Labeler
    prompts_labeler = get_prompts(outputs, raw_prompts)

    # Get AI feedback
    responses = asyncio.run(get_responses(prompts_labeler))