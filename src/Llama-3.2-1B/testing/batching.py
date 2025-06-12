"""
Batch Size and Token Generation Benchmark for Llama-3.2-1B

This script benchmarks the text generation performance of a local Llama-3.2-1B model
across different batch sizes and output token counts. It measures throughput (output tokens per second)
and total generation latency for each configuration, then plots the results for easy comparison.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import time
import matplotlib.pyplot as plt

logging.set_verbosity_error()  # Suppress logging messages

MODEL_PATH = "/home/gregdeli/greg_llms/models/Llama-3.2-1B"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

# Set the model to evaluation mode for inference
model.eval()

def test_batches(prompt, batch_sizes, num_tokens):
    """
    Runs generation benchmarks for various batch sizes and output token counts.
    Returns a list of dictionaries with throughput and latency results.
    """
    results = []
    for batch_size in batch_sizes:
        # Create a batch of identical prompts
        prompts = [prompt] * batch_size  
        inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        batch_results = {
            "batch_size": batch_size,
            "tokens_generated": [],
            "throughput": [],
            "latency": [],
        }

        for token_count in num_tokens:
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                # Generate token_count new tokens for each prompt in the batch
                ouputs = model.generate(
                    input_ids=input_ids, max_length=input_ids.shape[-1] + token_count
                )
            end_time = time.time()

            total_time = end_time - start_time
            combined_generated_tokens = token_count * batch_size
            throughput = combined_generated_tokens / total_time

            batch_results["tokens_generated"].append(token_count)
            batch_results["throughput"].append(throughput)
            batch_results["latency"].append(total_time)

            # Print the results for this configuration
            print(
                f"Batch Size: {batch_size}, Tokens Generated: {token_count}, Throughput: {throughput:.2f} output_tokens/sec, Latency (total generation time): {total_time:.2f}s"
            )

        results.append(batch_results)
        print()
    return results

# Define the batch sizes and output token counts to test
batch_sizes = [1, 2, 4, 8, 16, 32]
num_tokens = [25, 50, 100, 200]

prompt = "The key to life is"

# Run the benchmark and collect results
results = test_batches(prompt, batch_sizes, num_tokens)

def plot_results(results, metric, ylabel):
    """
    Plots the specified metric (throughput or latency) against the number of output tokens,
    for each batch size tested.
    """
    plt.figure(figsize=(10, 6))

    for batch_result in results:
        batch_size = batch_result["batch_size"]
        token_counts = batch_result["tokens_generated"]  # X-axis: token counts
        metric_values = batch_result[metric]  # Y-axis: metric values (throughput or latency)

        plt.plot(
            token_counts,
            metric_values,
            marker="o",
            linestyle="-",
            label=f"Batch Size = {batch_size}",
        )

    plt.xlabel("Output Tokens Generated")
    plt.ylabel(ylabel)  # "Throughputs" or "Latencies"
    plt.title(f"{metric.capitalize()} vs Tokens Generated for Different Batch Sizes")
    plt.legend(title="Batch Sizes")
    plt.grid(True)

    # Save the combined plot
    filepath = f"Plots/{metric}.png"
    plt.savefig(filepath)
    print(f"Saved plot: {filepath}")
    plt.close()


# Plot results
print("Plotting results...")
plot_results(results, "throughput", "Throughput (output_tokens/sec)")
plot_results(results, "latency", "Latency (total generation time in seconds)")
