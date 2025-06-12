"""
Forward Pass Profiling for Qwen1.5-MoE-A2.7B

This script profiles the latency and power consumption of each layer in the Qwen1.5-MoE-A2.7B model.
It uses Hugging Face Transformers and Accelerate for efficient model loading with disk offloading,
and measures both CPU/GPU timings and power usage for each layer during a forward pass.

* This script needs to be run with sudo privileges to access the power monitoring resources.
* The run_llm_python.sh script can be used to achieve this.

* The CPU/GPU timing measurements unfortunately capture disk I/O overhead as well.

Key steps:
- Loads the model with device/disk offloading.
- Registers hooks to capture timings, activations, and inputs for each layer.
- Runs a forward pass to collect profiling data.
- Measures power consumption for each layer using a custom PowerMonitor.
- Writes a detailed profiling report to a text file and plots the results.
"""

import torch
import time
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from power_monitor import PowerMonitor
from plot_results import plot_profiling_results

MODEL_PATH = "/home/gregdeli/greg_llms/models/Qwen1.5-MoE-A2.7B"

print("Loading model...")
config = AutoConfig.from_pretrained(MODEL_PATH)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(
    model, no_split_module_classes=["Qwen2MoeDecoderLayer"], dtype="float16"
)

# Manual disk offloading
for i in range(7, 24):  # Offload layers 7 through 23
    device_map[f"model.layers.{i}"] = "disk"
device_map.update({"model.norm": "disk", "model.rotary_emb": 0, "lm_head": "disk"})


OFFLOAD_DIR = "/home/gregdeli/greg_llms/.offload"
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

# Dictionaries to store profiling data
layer_inputs = {}  # Store input tensors for each layer
layer_modules = {}  # Store module references for each layer
activations = {}  # Store output activations for each layer
layer_timings = {}  # Store timing information for each layer
results = {}  # Store timing and power results for each layer

power_monitor = PowerMonitor()  # Initialize the power monitor


def get_layer_hooks(name):
    """
    Returns pre and post forward hooks for timing and activation capture for a given module name.
    """
    layer_timings[name] = {
        "host_start": None,
        "host_end": None,
        "start_event": None,
        "end_event": None,
    }

    def pre_hook(module, input):
        # Synchronize and record start time/events for timing
        torch.cuda.synchronize()
        layer_timings[name]["host_start"] = time.time()
        layer_timings[name]["start_event"] = torch.cuda.Event(enable_timing=True)
        layer_timings[name]["start_event"].record()

    def post_hook(module, input, output):
        # Record end time/events and calculate timings
        layer_timings[name]["end_event"] = torch.cuda.Event(enable_timing=True)
        layer_timings[name]["end_event"].record()
        torch.cuda.synchronize()
        layer_timings[name]["host_end"] = time.time()

        # Calculate GPU and CPU timings
        start_event = layer_timings[name]["start_event"]
        end_event = layer_timings[name]["end_event"]
        gpu_time = start_event.elapsed_time(end_event)  # Milliseconds

        host_start = layer_timings[name]["host_start"]
        host_end = layer_timings[name]["host_end"]
        cpu_time = (host_end - host_start) * 1000  # Convert to miliseconds

        results[name] = {
            "gpu_time": gpu_time,
            "cpu_time": cpu_time,
        }

        # Store input tensor for later reuse
        layer_inputs[name] = [t.detach() for t in input if t is not None]

        # Store module for later reuse
        layer_modules[name] = module

        # Store activations for later analysis
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach()

    return pre_hook, post_hook


# Register hooks to capture timings and activations, skipping specified module classes
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
    pre_hook, post_hook = get_layer_hooks(name)
    hooks.append(module.register_forward_pre_hook(pre_hook))
    hooks.append(module.register_forward_hook(post_hook))

# Input prompt
input_text = "The key to life is"
print("Tokenizing input...")
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Warmup runs
print("Warmup runs...")

# Warmup runs (optional, ensures the GPU is warmed up and CUDA kernels are loaded)
with torch.no_grad():
    for _ in range(1):
        model(**inputs)

print("Latency profiling run...")
with torch.no_grad():
    model(**inputs)

# Remove hooks
for hook in hooks:
    hook.remove()

# Power consumption profiling: repeat each layer's forward pass and measure power
NUM_REPEATS = 100  # Number of times to repeat each layer for averaging

print("Power consumption profiling runs...")
for name, module in layer_modules.items():
    # Get cached input for this layer
    args = [t.to("cuda") for t in layer_inputs[name]]
    args = tuple(args)

    power_monitor.start()
    print(f"Profiling {name} ({NUM_REPEATS} runs)...")
    for _ in range(NUM_REPEATS):
        with torch.no_grad():
            output = module(*args)
    power_monitor.stop()

    avg_gpu_power = power_monitor.get_avg_gpu_power()
    avg_cpu_power = power_monitor.get_avg_cpu_power()

    results[name].update({"avg_gpu_power": avg_gpu_power, "avg_cpu_power": avg_cpu_power})

# Write profiling results to a file
with open("Measurements/forward_pass_profiling.txt", "w") as f:
    f.write(
        f"{'Layer':<45} {'Output Shape':<30} {'GPU Latency (ms)':<20} {'CPU Latency (ms)':<20} {'GPU Power (mW)':<20} {'CPU Power (mW)':<20}\n"
    )
    f.write("=" * 155 + "\n")
    for layer_name in activations.keys():
        f.write(
            f"{layer_name:<45} {str(activations[layer_name].shape):<30} {results[layer_name]['gpu_time']:<20f} {results[layer_name]['cpu_time']:<20f} {results[layer_name]['avg_gpu_power']:<20f} {results[layer_name]['avg_cpu_power']:<20f}\n"
        )
print("Results written to 'Measurements/forward_pass_profiling.txt'")

# Sum up and print the execution times of all layers
total_gpu_time = sum(results[layer_name]["gpu_time"] for layer_name in results.keys())
total_cpu_time = sum(results[layer_name]["cpu_time"] for layer_name in results.keys())
print(f"Total GPU time: {total_gpu_time} ms")
print(f"Total CPU time: {total_cpu_time} ms")

# Cleanup to free memory
del inputs
torch.cuda.empty_cache()
