"""
Manual Token-by-Token Text Generation

This script demonstrates how to manually generate text from a prompt using a local Llama-3.2-1B model.
Instead of using the built-in `generate` method, it iteratively predicts and appends the next token,
allowing for fine-grained control over the generation process.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

# Input prompt
input_text = "Subject: Notification of Absence due to Illness\n\nDear [Boss's Name],"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Initialize the generated tokens with the input prompt tokens
generated_tokens = inputs["input_ids"]

# Set the model to evaluation mode for inference
model.eval()
with torch.no_grad():
    for _ in range(200):  # Limit to number of tokens generated
        # Forward pass: get logits for the next token
        outputs = model(input_ids=generated_tokens)
        next_token_logits = outputs.logits[:, -1, :]

        # Greedy decoding: select the token with the highest probability
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Append the predicted token to the sequence
        generated_tokens = torch.cat((generated_tokens, next_token_id), dim=-1)

        # Stop generation if the EOS (end-of-sequence) token is produced
        if next_token_id.item() == tokenizer.eos_token_id:
            break

# Decode the generated token IDs back to text
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)

# Print the generated text
print(f"\nGenerated Text: \n\n{generated_text}")
