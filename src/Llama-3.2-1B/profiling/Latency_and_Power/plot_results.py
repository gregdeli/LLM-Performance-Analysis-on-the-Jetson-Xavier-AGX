"""
Plotting Utilities for Layer-wise Profiling Results

This module provides functions to visualize latency and power profiling results
for each layer of the LLM. It generates bar plots for GPU/CPU latency
and power consumption, saving them as PNG and SVG files in the Measurements directory.

Functions:
    - plot_profiling_results: Plots both GPU/CPU latency and power for all layers.
    - plot_profiling_results_gpu_only: Plots only GPU latency and power for all layers.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_profiling_results(results):
    """
    Plots GPU and CPU latency, as well as GPU and CPU power consumption, for each layer.

    Args:
        results (dict): Dictionary mapping layer names to profiling metrics.
    """
    print("Plotting results...")

    # Extract layer names and metric values
    layers = list(results.keys())
    gpu_times = [results[layer]['gpu_time'] for layer in layers]
    cpu_times = [results[layer]['cpu_time'] for layer in layers]
    avg_gpu_powers = [results[layer]['avg_gpu_power'] for layer in layers]
    avg_cpu_powers = [results[layer]['avg_cpu_power'] for layer in layers]

    x = np.arange(len(layers))
    width = 0.2  # Bar width

    # Plot GPU and CPU Latency per layer
    fig_times, ax_times = plt.subplots(figsize=(max(12, len(layers)*0.4), 8))
    ax_times.bar(x - width/2, gpu_times, width, label='GPU Time (ms)')
    ax_times.bar(x + width/2, cpu_times, width, label='CPU Time (ms)')
    ax_times.set_xlabel('Layer')
    ax_times.set_title('GPU and CPU Latency per Layer')
    ax_times.set_xticks(x)
    ax_times.set_xticklabels(layers, rotation=90, fontsize=8)
    ax_times.legend()
    fig_times.tight_layout()
    plt.savefig("Measurements/latency_plot.png")
    plt.savefig("Measurements/latency_plot.svg")
    print("GPU and CPU Latency plot saved to 'Measurements/latency_plot'")

    # Plot GPU and CPU Power per layer
    fig_power, ax_power = plt.subplots(figsize=(max(12, len(layers)*0.4), 8))
    ax_power.bar(x - width/2, avg_gpu_powers, width, label='Avg GPU Power (mW)')
    ax_power.bar(x + width/2, avg_cpu_powers, width, label='Avg CPU Power (mW)')
    ax_power.set_xlabel('Layer')
    ax_power.set_title('GPU and CPU Power Consumption per Layer')
    ax_power.set_xticks(x)
    ax_power.set_xticklabels(layers, rotation=90, fontsize=8)
    ax_power.legend()
    fig_power.tight_layout()
    plt.savefig("Measurements/power_plot_gpu_only.png")
    plt.savefig("Measurements/power_plot_gpu_only.svg")
    print("GPU and CPU power plot saved to 'Measurements/power_plot'")


def plot_profiling_results_gpu_only(results):
    """
    Plots only GPU latency and GPU power consumption for each layer.

    Args:
        results (dict): Dictionary mapping layer names to profiling metrics.
    """
    print("Plotting results...")

    # Extract layer names and metric values
    layers = list(results.keys())
    gpu_times = [results[layer]['gpu_time'] for layer in layers]
    avg_gpu_powers = [results[layer]['avg_gpu_power'] for layer in layers]

    x = np.arange(len(layers))
    width = 0.35  # Bar width

    # Plot GPU Latency per layer
    fig_times, ax_times = plt.subplots(figsize=(max(12, len(layers)*0.4), 8))
    ax_times.bar(x, gpu_times, width, label='GPU Time (ms)')
    ax_times.set_xlabel('Layer')
    ax_times.set_title('GPU Latency per Layer')
    ax_times.set_xticks(x)
    ax_times.set_xticklabels(layers, rotation=90, fontsize=8)
    ax_times.legend()
    fig_times.tight_layout()
    plt.savefig("Measurements/latency_plot_gpu_only.png")
    plt.savefig("Measurements/latency_plot_gpu_only.svg")
    print("GPU Latency plot saved to 'Measurements/latency_plot'")

    # Plot GPU Power per layer
    fig_power, ax_power = plt.subplots(figsize=(max(12, len(layers)*0.4), 8))
    ax_power.bar(x, avg_gpu_powers, width, label='Avg GPU Power (mW)')
    ax_power.set_xlabel('Layer')
    ax_power.set_title('GPU Power Consumption per Layer')
    ax_power.set_xticks(x)
    ax_power.set_xticklabels(layers, rotation=90, fontsize=8)
    ax_power.legend()
    fig_power.tight_layout()
    plt.savefig("Measurements/power_plot_gpu_only.png")
    plt.savefig("Measurements/power_plot_gpu_only.svg")
    print("GPU power plot saved to 'Measurements/power_plot'")

    