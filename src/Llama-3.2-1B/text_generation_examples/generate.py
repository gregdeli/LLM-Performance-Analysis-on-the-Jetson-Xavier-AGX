"""
Direct Text Generation with Transformers

This script shows how to load a local Llama-3.2-1B model and tokenizer using Hugging Face Transformers,
generate a text completion for a given prompt, and print the result. The model is loaded with automatic
device placement and half-precision for efficient inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

config = AutoConfig.from_pretrained(MODEL_PATH)

# Load the model with automatic device placement and half-precision
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, device_map="auto", torch_dtype=torch.float16)

# Load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

prompt = "Subject: Notification of Absence due to Illness\n\nDear [Boss's Name],"
model_input = tokenizer([prompt], return_tensors="pt").to("cuda")

# Return generated tokens and convert to text
generated_ids = model.generate(
    **model_input,
    max_length=1024,
)

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"\nInput: {prompt}")
print(f"\nLlama-3.2-1B response:\n{output}")