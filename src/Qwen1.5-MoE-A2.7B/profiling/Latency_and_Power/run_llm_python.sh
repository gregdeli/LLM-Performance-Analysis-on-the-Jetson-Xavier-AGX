#!/bin/bash

# This script runs a Python script using the 'llm_env' Conda environment with sudo privileges.
# It is useful for running scripts that require elevated permissions, such as those that monitor
# system power usage and access hardware counters.
#
# Usage: ./run_llm_python.sh your_script.py 

sudo ~/greg_llms/miniconda3/envs/llm_env/bin/python "$@"