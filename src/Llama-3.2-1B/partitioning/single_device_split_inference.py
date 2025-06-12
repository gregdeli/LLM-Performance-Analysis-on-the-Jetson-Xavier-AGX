"""
Single-Device Model Split Inference for Llama-3.2-1B

This script demonstrates how to manually split a Llama-3.2-1B model into two sequential
submodules (first half and second half) on a single device. The script defines custom
PyTorch modules for each half, runs inference through both halves, and decodes the output.

Adjust the SPLIT_INDEX to change the split point between the two halves.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

# Load the full model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Split point configuration (adjust as needed)
SPLIT_INDEX = 8  # Split after 8th transformer layer 

class FirstHalf(torch.nn.Module):
    """
    Module representing the first half of the model: embeddings + first N transformer layers.
    """
    def __init__(self, original_model):
        super().__init__()
        self.embed_tokens = original_model.model.embed_tokens
        self.layers = torch.nn.ModuleList(original_model.model.layers[:SPLIT_INDEX])

        self.device = original_model.device # Preserve original device

    def forward(self, input_ids):
        # Generate position ids for the input sequence
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)

        # Forward pass through embeddings and first half of transformer layers
        tok_embeds = self.embed_tokens(input_ids)
        x = tok_embeds
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)[0]
        return x
        
class SecondHalf(torch.nn.Module):
    """
    Module representing the second half of the model: remaining transformer layers, norm, and output head.
    """
    def __init__(self, original_model):
        super().__init__()
        self.layers = torch.nn.ModuleList(original_model.model.layers[SPLIT_INDEX:])
        self.norm = original_model.model.norm
        self.lm_head = original_model.lm_head

        self.device = original_model.device # Preserve original device

    def forward(self, hidden_states):
        # Generate position_ids for the second half
        seq_length = hidden_states.shape[1]
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)

        # Forward pass through remaining transformer layers, norm, and output head
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids=position_ids)[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    

# Instantiate the split models and move them to the appropriate device
first_half_model = FirstHalf(model).to(model.device).half() # Use half precision
second_half_model = SecondHalf(model).to(model.device).half() # Use half precision

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Run inference on the first half of the model
with torch.no_grad():
    first_half_output = first_half_model(inputs.input_ids)

# Run inference on the second half of the model
with torch.no_grad():
    logits = second_half_model(first_half_output)

# Decode the output
predicted_token = torch.argmax(logits[0, -1]).item()
print(f"\nGenerated text: {prompt}{tokenizer.decode(predicted_token)}")
        
        