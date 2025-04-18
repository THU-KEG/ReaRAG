#!/usr/bin/env bash

# Exit script on any error
set -e

# Source config to load environment variables
source deploy_config.sh

# Optionally, create a logs directory so everything is tidy
mkdir -p logs

# ################################### 1. DEPLOY RETRIEVER (FOR RAG Engine) ###################################
echo "Starting Retriever..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_RETRIEVER nohup python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever 1 \
    --port $RETRIEVER_PORT \
  > logs/retriever.log 2>&1 &

sleep 5
# ################################### 2. DEPLOY ANSWER LLM (FOR RAG Engine) ###################################
echo "Starting ANSWER LLM..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_ANSWER_LLM nohup python vllm_async_serving.py \
    --model $ANSWER_MODEL_DIR \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.6 \
    --host $ANSWER_LLM_HOST \
    --port $ANSWER_LLM_PORT \
    --trust-remote-code \
  > logs/answer_llm.log 2>&1 &
sleep 5

#### To kill the processes, you can use the following command:
# ps aux | grep vllm_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep retriever_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep vllm_async_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
