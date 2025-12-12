from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import asyncio
from openai import AsyncOpenAI
import numpy as np

# Parameters
model = "Qwen/Qwen3-1.7B"
labeler_model = "Qwen/Qwen3-14B"
n_prompts = 2
n_responses_per_prompt = 16
init_seed = 16
mu = 1e-4
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

async def request(prompt, client):
    response = await client.chat.completions.create(
        model=labeler_model,
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

async def get_responses(prompts, client):
    tasks = [request(prompt, client) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

def get_answer(response):
    try:
        start = response.find("\\boxed{") + 7
        assert response[start] in ["A", "B"]
        if response[start] == "A":
            return 0
        else:
            return 1
    except Exception:
        return None

def get_preference(responses):
    answers = []
    for r in responses:
        answer = get_answer(r)
        if answer is not None:
            answers.append(answer)
            
    proba = np.array(answers).mean()
    return proba

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# Configure the sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=1, n=n_responses_per_prompt, max_tokens=8192)

# Initialize the vLLM engine with worker extension
vllm_model = LLM(model=model, worker_extension_cls="utils.myworker.MyWorker")

# Prepare the dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:4000]")
train_loader = DataLoader(dataset=dataset["prompt"], batch_size=n_prompts, collate_fn=collate_fn, shuffle=False, pin_memory=True)

async def main():
    # Create client / AI Labeler
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
        
    for step, (raw_prompts, formatted_prompts) in enumerate(train_loader):
        seed = init_seed + step
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
        responses = await get_responses(prompts_labeler, client)

        # Get preference probability Pr[1>0]
        proba = get_preference(responses)

        # Update parameters based on AI labeler preference
        if proba > 1/2:
            vllm_model.collective_rpc("perturb_params", args=(-mu, seed))
        elif proba < 1/2:
            vllm_model.collective_rpc("perturb_params", args=(mu, seed))

if __name__ == "__main__":
    asyncio.run(main())