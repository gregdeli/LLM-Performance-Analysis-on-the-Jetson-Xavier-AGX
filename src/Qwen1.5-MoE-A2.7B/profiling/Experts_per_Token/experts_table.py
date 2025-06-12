"""
Expert Activations Table Visualization

* experts_per_token.py needs to be run first to generate the expert activations data.

This script loads the expert activation data (top-k experts per token per layer) from a JSON file
and visualizes it as a table using Matplotlib. Each row corresponds to a model layer, each column
to a token, and each cell lists the indices of the most activated experts for that token at that layer.

- Loads expert activation data from Measurements/expert_activations.json.
- Prepares a table where rows are layers and columns are tokens.
- Each cell contains the top-k expert indices for that token/layer.
- Saves the visualization as Measurements/expert_activation_table.png.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the expert activations data
json_file_path = "Measurements/expert_activations.json"  # Or your actual path
with open(json_file_path, "r") as f:
    expert_data = json.load(f)

# Prepare data for the table: layers as rows, tokens as columns
layer_indices = sorted(expert_data.keys(), key=int)

# Get the list of tokens from the first layer (assumes all layers have the same tokens)
tokens = [token for token in expert_data[layer_indices[0]].keys()]

cell_texts = []
for layer_idx_str in layer_indices:
    row_texts = []
    for token in expert_data[layer_indices[0]].keys():
        experts = expert_data[layer_idx_str].get(token, [])
        row_texts.append(", ".join(map(str, experts)))
    cell_texts.append(row_texts)

# Create the matplotlib figure
fig, ax = plt.subplots(figsize=(len(tokens) * 2, len(layer_indices) * 0.3))  # Adjust size
ax.axis("tight")
ax.axis("off")  # Hide the axes for a clean table look

# Add the table to the plot
table = ax.table(
    cellText=cell_texts,
    rowLabels=[f"Layer {l}" for l in layer_indices],
    colLabels=tokens,
    loc="center",
    cellLoc="center",
)

# Adjust table appearance
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust cell size

plt.title("Expert Activations per Token per Layer", fontsize=14, pad=20)
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.savefig("Measurements/expert_activation_table.png", dpi=300, bbox_inches="tight")

print("Expert activation table saved to Measurements/expert_activation_table.png")
