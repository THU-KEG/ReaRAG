# Deployment

## 🔧 Scripts Introduction

The table below describes each `.sh` file in the `deploy/` directory:

| Script Name       | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `deploy_rag_engine.sh`       | Deploys the retriever and a language model for answer generation based on retrieved documents. The LLM is served via the VLLM async server.                       |
| `deploy_async.sh`        | Launches the VLLM async server. |
| `deploy.sh`      | Launches the VLLM server (non-async version).                   |
| `deploy_config.sh` | Configuration file used by `deploy_rag_engine.sh` and `deploy.sh`.                             |

For detailed setup instructions, please refer to the [main README](../README.md).

## 🛑 Stopping services
To stop running services, use the commands below according to what you launched:
```
# Kill process running 'vllm_serving.py'
ps aux | grep vllm_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9

# Kill process running 'retriever_serving.py'
ps aux | grep retriever_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9

# Kill process running 'vllm_async_serving.py'
ps aux | grep vllm_async_serving.py | grep -v grep | awk '{print $2}' | xargs kill -9
```

