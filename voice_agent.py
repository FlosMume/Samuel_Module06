# voice_agent.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from tools import search_arxiv, calculate
from prompts import SYSTEM_PROMPT
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv("C:/Users/ch939/Downloads/LLMBootCampCodes/Week5/.env")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    token=HF_TOKEN,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
)

# Tool registry
TOOLS = {
    "search_arxiv": search_arxiv,
    "calculate": calculate,
}

def route_llm_output(llm_output: str) -> str:
    """Parse LLM output and route to tools."""
    llm_output = llm_output.strip()
    try:
        # Try to parse as JSON
        call = json.loads(llm_output)
        func_name = call.get("function")
        args = call.get("arguments", {})

        if func_name in TOOLS:
            func = TOOLS[func_name]
            return func(**args)
        else:
            return f"Error: Unknown function '{func_name}'"
    except json.JSONDecodeError:
        # Not a function call — return as-is
        return llm_output

def agent_query(user_text: str) -> str:
    """Main agent loop."""
    # prompt = f"<|begin_of_sentence|>System: {SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"
    prompt = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    response = pipe(
        prompt,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id   
    )[0]["generated_text"]
    # Extract only assistant response
    assistant_reply = response[len(prompt):].strip()

    # Route and return
    final_reply = route_llm_output(assistant_reply)
    return final_reply

# Test
print(tokenizer.special_tokens_map) # Should show: {'eos_token': '<|eot_id|>'}
if __name__ == "__main__":
    print(agent_query("What is 5 + 3 * 2?"))
    print(agent_query("Search for recent papers on quantum entanglement."))
    print(agent_query("Hello, how are you?"))