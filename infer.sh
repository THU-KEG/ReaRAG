#!/usr/bin/env bash

python -m src.infer \
    --agent_api http://localhost:9891/generate \
    --retriever_api http://localhost:9892/search \
    --gen_api http://localhost:9893/generate \
    --rearag_tokenizer_path THU-KEG/ReaRAG-9B \
    --ans_tokenizer_path ZhipuAI/glm-4-9b-chat

