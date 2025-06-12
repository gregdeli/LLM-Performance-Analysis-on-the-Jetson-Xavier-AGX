"""
Example: Simple Text Generation Pipeline

This script demonstrates how to use Hugging Face's pipeline utility to load a local
Llama-3.2-1B model and tokenizer for text generation. It generates a completion
for a given input prompt and prints the result to the terminal.
"""

import torch
from transformers import pipeline

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

# Initialize the text generation pipeline
generator = pipeline("text-generation", 
                     model=MODEL_PATH, 
                     tokenizer=MODEL_PATH, 
                     torch_dtype=torch.float16,
                     device_map="auto", )

# Generate text
input_text = "Computer Engineering is a field that involves"
generated_text = generator(input_text, max_length=128)

# Print the output on the terminal
print(f"\nInput prompt:\n{input_text}")
print(f"\nLlama 3.2 1B response:\n{generated_text[0]['generated_text']}")
