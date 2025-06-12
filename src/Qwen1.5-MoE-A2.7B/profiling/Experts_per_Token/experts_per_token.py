"""
Experts Activated Per Token Profiler

This script profiles which experts are activated for each token at each layer in the Qwen1.5-MoE-A2.7B model.
It uses Hugging Face Transformers and Accelerate to efficiently load the model with disk offloading,
registers forward hooks to capture gate activations, and generates a single token from a prompt.
The script then extracts and saves the top-4 experts (by gate probability) per token per layer to JSON and text files.

Key steps:
- Loads the model with device/disk offloading for memory efficiency.
- Registers hooks to capture gate activations during inference.
- Runs a short generation loop.
- Extracts and saves the top-k experts per token per layer.
"""

import re
import torch
import json
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/gregdeli/greg_llms/models/Qwen1.5-MoE-A2.7B"

print("Loading model...")
config = AutoConfig.from_pretrained(MODEL_PATH)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(
    model, no_split_module_classes=["Qwen2MoeDecoderLayer"], dtype="float16"
)

# Manual disk offloading
for i in range(11, 24):  # Offload layers 11 through 23 to disk
    device_map[f"model.layers.{i}"] = "disk"
device_map.update({"model.norm": "disk", "model.rotary_emb": 0, "lm_head": "disk"})

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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

activations = {}


def get_layer_hooks(name):
    """
    Returns a forward hook that stores the output activations for the given module name.
    """

    def post_hook(module, input, output):
        # Store activations for later analysis
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach()

    return post_hook


# Register hooks to capture activations, skipping specified module classes
SKIP_CLASSES = {
    "Qwen2MoeForCausalLM",
    "Qwen2MoeModel",
    "ModuleList",
    "Qwen2MoeDecoderLayer",
    "Qwen2MoeAttention",
    "Qwen2MoeSparseMoeBlock",
    "Qwen2MoeMLP",
}

hooks = []

for name, module in model.named_modules():
    # Skip modules whose class name matches skip list
    if module.__class__.__name__ in SKIP_CLASSES:
        continue
    post_hook = get_layer_hooks(name)
    hooks.append(module.register_forward_hook(post_hook))

# Input prompt
input_text = "The meaning"
print("Tokenizing input...")
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

generated_tokens = inputs["input_ids"]

print("Running inference...")
with torch.no_grad():
    # Generate a token
    outputs = model(input_ids=generated_tokens)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    generated_tokens = torch.cat((generated_tokens, next_token_id), dim=-1)

# Remove hooks after inference
for hook in hooks:
    hook.remove()

# Decode the output to text
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
print(f"Qwen-1.5-MoE: {generated_text}")

# Get the input tokens to map activations
input_ids = inputs.input_ids[0].cpu().tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("Capturing the Activated Experts per Token per Layer...")
gate_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate")

topk_by_dec_layer = {}

# Extract top-4 experts per token per layer from the captured activations
for layer_name in activations.keys():
    gate_match = gate_pattern.fullmatch(layer_name)

    if gate_match:
        layer_idx = gate_match.group(1)

        # Get the topk experts for this layer
        gate_output = activations[layer_name]  # shape: [batch_size, num_experts]
        probs = torch.softmax(gate_output, dim=-1)
        topk = torch.topk(probs, k=4, dim=-1)  # shape: [batch_size, 4]

        layer_dict = {}
        for token_pos, token in enumerate(tokens):
            experts = topk.indices[token_pos].tolist()
            layer_dict[token] = experts

        topk_by_dec_layer[layer_idx] = layer_dict

# Save topk_by_dec_layer as a json file
json_file_path = "Measurements/expert_activations.json"
with open(json_file_path, "w") as f:
    json.dump(topk_by_dec_layer, f, indent=4)
print(f"Expert activations saved to {json_file_path}")

# Save results to a human-readable text file
with open("Measurements/expert_activations.txt", "w") as f:
    for layer_idx in topk_by_dec_layer.keys():
        f.write(f"Layer {layer_idx}:\n")
        for token, experts in topk_by_dec_layer[layer_idx].items():
            experts_str = ", ".join(map(str, experts))
            f.write(f"  {token}: [{experts_str}]\n")
        f.write("\n")
print("Expert activations saved to Measurements/expert_activations.txt")


# Cleanup to free memory
del inputs
torch.cuda.empty_cache()
