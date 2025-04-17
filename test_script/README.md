# Test script Readme
This directory contains test script to test the deployment of:
1. Retriever. Deployed with `ReaRAG/deploy/retriever_serving.py`. Test with `test_retriever.py`
2. VLLM server. Deployed with `ReaRAG/deploy/vllm_serving.py`. Test with `test_vllm.py`
3. VLLM async server. Deployed with `ReaRAG/deploy/vllm_async_serving.py`. Test with `test_vllm_async.py`

Note:
1. We assume QwQ is deployed with `test_vllm_async.py`.   
2. Modify the config in `test_config.yaml`