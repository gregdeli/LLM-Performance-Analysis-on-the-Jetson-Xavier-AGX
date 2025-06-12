#!/bin/bash

# This script launches a Python process from the 'llm_env' Conda environment
# with remote debugging enabled via debugpy. It listens for a debugger to attach
# on port 5678 and waits until a client connects before starting execution.
# Usage: ./debug_llm_python.sh your_script.py 

# After running this script, you can attach a debugger (like VS Code)
# to the specified port (5678) to debug the Python script.
# * See .vscode/launch.json for configuration details.

sudo /home/gregdeli/greg_llms/miniconda3/envs/llm_env/bin/python -m debugpy \
--listen 5678 \
--wait-for-client \
"$@"