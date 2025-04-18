# ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation

<p align="center">
   🤗 <a href="https://huggingface.co/THU-KEG/ReaRAG-9B" target="_blank">Model</a> • 🤗 <a href="https://huggingface.co/datasets/THU-KEG/ReaRAG-20k" target="_blank">Dataset</a> • 📃 <a href="https://arxiv.org/abs/2503.21729" target="_blank">Paper</a>
</p>

## 🔍 Table of Contents
- [📜 Introduction](#introduction)
- [⚙️ Environment setup](#environment)
- [🔨 Data construction](#data)
- [🏋🏻‍♂️ ReaRAG Training](#training)
- [🤖️ Inference](#inference)
- [📝 Citation](#citation)

<a name="introduction"></a>
## 📜  Introduction
<p align="center">
  <img src="figs/rearag_overview.png" alt="ReaRAG Overview" width="600">
</p>
Large Reasoning Models (LRMs) exhibit remarkable reasoning abilities but rely primarily on parametric knowledge, which limits their factual accuracy. To address this limitation, we propose ReaRAG, a factuality enhanced reasoning model that iteratively constructs reasoning chains guided by knowledge retrieval, while efficiently exploring diverse queries without excessive iterations.

<a name="environment"></a>
## ⚙️ Environment setup
### Conda environment

```
conda create --name rearag python=3.10 -y && conda activate rearag

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install vllm==0.6.5

pip install datasets flask langid uvicorn termcolor jieba fuzzywuzzy rouge

conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
### RAG engine environment
#### Preliminaries
Please note that the RAG engine provided here differs from the implementation described in the original paper. To facilitate public usage, we offer a practical and simplified version in this repository.  

To setup RAG engine, first download the following:
* E5-base-v2 Retriever. 🤗 [Link](https://huggingface.co/intfloat/e5-base-v2)
* 2018 Wikipedia Corpus. 🤗 [Link](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus)
* Indexed 2018 Wikipedia Corpus. [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip)

<a name="rag_engine_deployment"></a>
####  Deployment steps:
1. Modify the config in `ReaRAG/deploy/deploy_config.sh` and `ReaRAG/deploy/retriever_config.yaml`
2. Run the deployment script, make sure you see the phrase `'xx running on http://{host}:{port}'` to confirm deployment:
```
# From within ReaRAG/deploy/
bash deploy_rag_engine.sh
```

<a name="data"></a>
## 🔨 Data construction
### Deployment step:
Before starting, make sure you have deployed the `rag_engine` (see [RAG Engine Deployment](#rag_engine_deployment)). Then, follow the steps below to deploy a LLM/LRM (e.g., `QwQ-32b-preview`) for data construction:

1. Modify the environment variables in `ReaRAG/deploy/deploy_async.sh`.
2. Run the deployment script, make sure you see the phrase `'xx running on http://{host}:{port}'` to confirm deployment:
```
# From within ReaRAG/deploy/
bash deploy_async.sh
```

### Begin data construction:
We construct data from HotpotQA, MuSiQue, and NQ. Therefore, make sure you have downloaded them and processed them into following structure:
```json
{
  "question": "What is the capital of ...",
  "answer": "The capital of xxx is ...",
}
```

Next, modify the config in `ReaRAG/src_data/data_config.yaml`. Then, execute script below:
```
# From within ReaRAG/
bash data_construct.sh
```

The result of the data will be saved at `ReaRAG/src_data/data`, named `conv_qwq.json` for example, where each data is a list of conversation, structured as below:
```json
{
    "messages": [{"role": "user", "content": "..."}, 
                 {"role": "assistant", "reasoning": "..."},
                 {"role": "observation", "content": "..."}, ...]
}
```
During sft, the loss is computed only on messages that contain the `reasoning` key, rather than the `content` key.

<a name="training"></a>
## 🏋🏻‍♂️ Training
Training data can be found from ([🤗 huggingface](https://huggingface.co/datasets/THU-KEG/ReaRAG-20k)).  
You can mix it with general SFT data such as [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset). We adopt [Metragon-LM](https://github.com/NVIDIA/Megatron-LM) for model training. For a more lightweight implementation, you may adopt the code and environment from [LongAlign](https://github.com/THUDM/LongAlign).


<a name="inference"></a>
## 🤖️ Inference
### Deployment step:
Before starting, make sure you have deployed the `rag_engine` (see [RAG Engine Deployment](#rag_engine_deployment)). Then, follow the steps below to deploy 
ReaRAG:
1. Modify the config in `ReaRAG/deploy/deploy_config.sh`.
2. Run the deployment script, make sure you see the phrase `'xx running on http://{host}:{port}'` to confirm deployment:
```
# From within ReaRAG/deploy/
bash deploy.sh 
```

### Begin usage:
Next, modify the config in `ReaRAG/infer.sh`. Then, execute script below:
```
# From within ReaRAG/
bash infer.sh
```

<a name="citation"></a>
## 📝 Citation
If you find our work useful, please consider citing ReaRAG:
```
@article{lee2025rearag,
  title={ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation},
  author={Lee, Zhicheng and Cao, Shulin and Liu, Jinxin and Zhang, Jiajie and Liu, Weichuan and Che, Xiaoyin and Hou, Lei and Li, Juanzi},
  journal={arXiv preprint arXiv:2503.21729},
  year={2025}
}
```

