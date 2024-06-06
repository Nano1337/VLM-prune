#!/bin/bash

# This Bash script runs the Python script with arguments

# Run the Python script with command-line arguments
python vlm_layer_similarity.py --model_path "clip-flant5-xxl" \
                      --batch_size 17 \
                      --max_length 1024 \
                      --layers_to_skip 10 \
                      --dataset_size 10000 \
                      --dataset_subset "train" 