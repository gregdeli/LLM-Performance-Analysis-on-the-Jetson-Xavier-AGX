"""
Streaming Text Generation.

This script demonstrates how to load a local Llama-3.2-1B model and tokenizer,
generate text from a prompt, and stream the output tokens live to the console
using Transformers' TextStreamer utility.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

# Load the model with automatic device placement
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)

# Load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

prompt = "Subject: Notification of Absence due to Illness\n\nDear [Boss's Name],"

# Tokenize the prompt and move tensors to the GPU
model_input = tokenizer([prompt], return_tensors="pt").to("cuda")

# Create a TextStreamer to stream generated tokens to the console as they are produced
streamer = TextStreamer(tokenizer)

# Generate up to 100 new tokens, streaming output live to the console
_ = model.generate(**model_input, streamer=streamer, max_new_tokens = 100)