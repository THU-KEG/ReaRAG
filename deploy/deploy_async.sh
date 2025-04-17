#!/usr/bin/env bash

# Exit script on any error
set -e

# Environment variables
MODEL_DIR="Qwen/QwQ-32B-Preview"
GPU=4,5
PORT=9894
HOST="127.0.0.1"

# Optionally, create a logs directory so everything is tidy
mkdir -p logs

# ################################### 1. DEPLOY LRM/LLM for data construction ###################################
echo "Starting deploying LRM/LLM for data construction..."
CUDA_VISIBLE_DEVICES=$GPU nohup python vllm_async_serving.py \
  --model $MODEL_DIR \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.8 \
  --trust-remote-code \
  --host $HOST \
  --port $PORT \
  > logs/vllm_async_serving.log 2>&1 &
sleep 5

#### To kill the processes, you can use the following command:
# ps aux | grep vllm_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep retriever_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep vllm_async_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
