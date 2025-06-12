"""
This script demonstrates how to use a local Llama-3.2-1B-Instruct model for text generation
with the Hugging Face Transformers pipeline.

It shows how to structure a conversational prompt
by specifying roles (such as 'system' and 'user') in the messages list. The model generates
a response based on these roles, allowing for flexible multi-turn dialogue setups.
"""

import torch
from transformers import pipeline

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B-Instruct"

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Define the conversation messages for the prompt
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you help me write an email to my boss calling in sick?"},
]

# Generate a response from the model
outputs = pipe(
    messages,
    max_new_tokens=1024,
)

# Print the prompts and the model's generated response
print(f"\nSystem Prompt: {messages[0]['content']}")
print(f"\nUser Prompt: {messages[1]['content']}")
print(f"\nLlama-3.2-1B response:\n{outputs[0]['generated_text'][2]['content']}\n")
