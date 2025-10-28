#!/bin/bash

# Vision-Prompt Glasses - Conda Environment Runner
echo "Starting Vision-Prompt Glasses in conda environment..."

# Activate conda environment and run the application
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv
python main.py
