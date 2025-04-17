"""
This script tests the API deployed with ReaRAG/deploy/vllm_serving.py
"""

from transformers import AutoTokenizer
import requests
import yaml

with open('src_data/test_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['vllm']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], trust_remote_code=True)
end_token = tokenizer.eos_token

user_input = "What are the benefits of using Retrieval-Augmented Generation?"
prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=True
        )
print(f"Prompt:\n{prompt}")

# Define the input and parameters
data = {
    "inputs": prompt,
    "parameters": {
        "max_tokens": 1048,
        "top_p": 0.9,
        "temperature": 0.7,
        "stop": [end_token],
        "skip_special_tokens": False
    }
}

# Send to Flask endpoint
response = requests.post(config['api'], json=data)

# Print result
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
