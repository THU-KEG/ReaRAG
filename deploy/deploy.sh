#!/usr/bin/env bash

# Exit script on any error
set -e

# Source config to load environment variables
source deploy_config.sh

# Optionally, create a logs directory so everything is tidy
mkdir -p logs

# ################################### 1. DEPLOY REARAG ###################################
echo "Starting REARAG..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_REARAG nohup python vllm_serving.py \
  --model_path $REARAG_MODEL_DIR \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.6 \
  --host $REARAG_HOST \
  --port $REARAG_PORT \
  > logs/rearag.log 2>&1 &
sleep 5

#### To kill the processes, you can use the following command:
# ps aux | grep vllm_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep retriever_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep vllm_async_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9



