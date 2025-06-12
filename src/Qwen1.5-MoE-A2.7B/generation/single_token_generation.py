"""
Qwen1.5-MoE-A2.7B Generation Example

This script demonstrates how to load and generate text with the Qwen1.5-MoE-A2.7B model
using Hugging Face Transformers and Accelerate.
It shows how to:
- Set up device mapping for model offloading (including disk offloading, which is necessary due to the model's size).
- Load the model and tokenizer,
- Perform a single-step text generation.

The script is optimized for limited memory environments.
"""

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/home/gregdeli/greg_llms/models/Qwen1.5-MoE-A2.7B"

# Load model configuration
config = AutoConfig.from_pretrained(MODEL_PATH)

# Initialize model weights without allocating memory
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Automatically infer device map, specifying which modules should not be split
device_map = infer_auto_device_map(
    model, no_split_module_classes=["Qwen2MoeDecoderLayer"], dtype="float16"
)

# Manually offload specific layers to disk to save memory
for i in range(7, 24):  # Offload layers 7 through 23
    device_map[f"model.layers.{i}"] = "disk"
device_map.update(
    {
        "model.norm": "disk",
        "model.rotary_emb": 0,
        "lm_head": "disk",
    }  # The rotary embedding layer is small enough to stay on memory
)


OFFLOAD_DIR = "/home/gregdeli/greg_llms/.offload"

# Load the model with the specified device map and offloading options
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=device_map,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

# Input prompt
input_text = "Athens is the capital of"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# Clean memory before generating
torch.cuda.empty_cache()

model.eval()
with torch.no_grad():
    # Forward pass: get logits for the next token
    outputs = model(input_ids=inputs["input_ids"])
    next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token position
    next_token_id = torch.argmax(next_token_logits, dim=-1)  # Select the most probable next token

    # Concatenate input tokens with the generated token
    all_token_ids = torch.cat([inputs["input_ids"][0], next_token_id])

# Decode the full sequence (input + generated token) to text
output_text = tokenizer.decode(all_token_ids)
print(f"\nOutput: \n{output_text}")

# Memory cleanup
del inputs, outputs
torch.cuda.empty_cache()
