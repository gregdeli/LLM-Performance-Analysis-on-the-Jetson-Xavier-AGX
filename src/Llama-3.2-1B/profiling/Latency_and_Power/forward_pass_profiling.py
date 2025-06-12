"""
Layer-wise Latency and Power Profiling for Llama-3.2-1B

This script profiles the forward pass of a local Llama-3.2-1B model at the layer level.
It measures both GPU and CPU latency for each layer using hooks, and then benchmarks
the average GPU and CPU power consumption for each layer by repeatedly executing them.
Results are written to a file and can be visualized with the included plotting utility.

* This script needs to be run with sudo privileges to access the power monitoring resources.
* The run_llm_python.sh scipt can be used to achieve this.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_monitor import PowerMonitor
from plot_results import plot_profiling_results


MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

print("Loading model...")
# Load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)

print("Loading tokenizer...")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

# Dictionaries to store intermediate data for each layer
layer_inputs = {}    # Stores input tensors for each layer
layer_modules = {}   # Stores module references for each layer
activations = {}     # Stores output activations for each layer
layer_timings = {}   # Stores timing info for each layer
results = {}         # Stores final profiling results

# Initialize the power monitor utility
power_monitor = PowerMonitor()

def get_layer_hooks(name):
    """
    Returns pre- and post-forward hooks for timing a layer's execution.
    Hooks record both CPU (host) and GPU (CUDA) timing events.
    """
    layer_timings[name] = {
        'host_start': None,
        'host_end': None,
        'start_event': None,
        'end_event': None
    }

    def pre_hook(module, input):
        torch.cuda.synchronize()
        layer_timings[name]['host_start'] = time.time()
        layer_timings[name]['start_event'] = torch.cuda.Event(enable_timing=True)
        layer_timings[name]['end_event'] = torch.cuda.Event(enable_timing=True)
        layer_timings[name]['start_event'].record()
        
    def post_hook(module, input, output):
        layer_timings[name]['end_event'].record()
        torch.cuda.synchronize()
        layer_timings[name]['host_end'] = time.time()

        # Calculate Timings
        # GPU time
        start_event = layer_timings[name]['start_event']
        end_event = layer_timings[name]['end_event']
        gpu_time = start_event.elapsed_time(end_event)  # Milliseconds

        # CPU time
        host_start = layer_timings[name]['host_start']
        host_end = layer_timings[name]['host_end']
        cpu_time =  (host_end - host_start) * 1000 # Convert to milliseconds

        results[name] = {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
        }

        # Store input tensor for later reuse
        layer_inputs[name] = [t.detach() for t in input if t is not None]

        # Store module for later reuse
        layer_modules[name] = module

        # Store activations for output shape reporting
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach()
    
    return pre_hook, post_hook

# Register hooks
SKIP_CLASSES = {
    "LlamaForCausalLM",  
    "LlamaModel",  
    "ModuleList",  
    "LlamaDecoderLayer",  
    "LlamaAttention",
    "LlamaMLP"  
}

hooks = [] # Store hooks to remove later

# Register hooks on all layers except those in SKIP_CLASSES
for name, module in model.named_modules():
    if module.__class__.__name__ in SKIP_CLASSES:
        continue
    pre_hook, post_hook = get_layer_hooks(name)
    hooks.append(module.register_forward_pre_hook(pre_hook))
    hooks.append(module.register_forward_hook(post_hook))

# Input prompt
input_text = "The key to life is"
print("Tokenizing input...")
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Warmup runs to ensure CUDA kernels are loaded and GPU is ready
print("Warmup runs...")
with torch.no_grad():
    for _ in range(5):
        model(**inputs)

# Run a single forward pass to profile latency for each layer
print("Latency profiling run...")
with torch.no_grad():
    model(**inputs)

# Remove hooks after profiling
for hook in hooks:
    hook.remove()

# Power consumption profiling: repeat each layer's forward pass and measure power
NUM_REPEATS = 100 #Number of times to repeat each layer

print("Power consumption profiling runs...")
for name, module in layer_modules.items():
    # Get cached input for this layer and move to GPU
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

    results[name].update({
        'avg_gpu_power': avg_gpu_power,
        'avg_cpu_power': avg_cpu_power
    })

torch.cuda.empty_cache() # Clear CUDA cache to free memory

# Write profiling results to a file 
with open("Measurements/forward_pass_profiling.txt", "w") as f:
    f.write(f"{'Layer':<45} {'Output Shape':<30} {'GPU Latency (ms)':<20} {'CPU Latency (ms)':<20} {'GPU Power (mW)':<20} {'CPU Power (mW)':<20}\n")
    f.write("=" * 155 + "\n")
    for layer_name in results.keys():
        f.write(f"{layer_name:<45} {str(activations[layer_name].shape):<30} {results[layer_name]['gpu_time']:<20f} {results[layer_name]['cpu_time']:<20f} {results[layer_name]['avg_gpu_power']:<20f} {results[layer_name]['avg_cpu_power']:<20f}\n")
print("Results written to 'Measurements/forward_pass_profiling.txt'")

# Print total execution times for all layers
total_gpu_time = sum(results[layer_name]['gpu_time'] for layer_name in results.keys())
total_cpu_time = sum(results[layer_name]['cpu_time'] for layer_name in results.keys())
print(f"Total GPU time: {total_gpu_time} ms")
print(f"Total CPU time: {total_cpu_time} ms")

# Plot the profiling results (requires plot_results.py)
plot_profiling_results(results)

