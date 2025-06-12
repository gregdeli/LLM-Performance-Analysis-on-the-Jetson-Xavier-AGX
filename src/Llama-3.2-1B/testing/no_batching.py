"""
Single-Prompt Text Generation Benchmark

This script benchmarks the text generation performance of a local Llama-3.2-1B model
for a single input prompt (no batching). It measures throughput (output tokens per second)
and total generation latency for a specified number of new tokens, then prints the results
along with the generated text.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

# Load the model 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

# Set the model to evaluation mode for inference
model.eval()

def measure_time(prompt, max_new_tokens=100):
    """
    Generates text for a single prompt and measures throughput and latency.

    Returns:
        throughput: Output tokens per second
        total_time: Total generation time in seconds
        generated_tokens: Number of new tokens generated
        output: Generated token IDs
    """
    input = tokenizer([prompt], return_tensors="pt").to("cuda")
    start_time = time.time()
    # Generate text with the specified number of new tokens
    output = model.generate(**input, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate tokens generated and time taken
    generated_tokens = len(output[0]) - len(input["input_ids"][0])
    total_time = end_time - start_time
    throughput = generated_tokens / total_time

    return throughput, total_time, generated_tokens, output


prompt = "The key to life is"
tps, total_time, tokens, output = measure_time(prompt)
print(
    f"\nThroughput: {tps:.2f} output tokens / sec, Latency (total generation time): {total_time:.2f}s, Tokens generated: {tokens}"
)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"\nInput: {prompt}")
print(f"\nLlama-3.2-1B response:\n{decoded_output}\n")
