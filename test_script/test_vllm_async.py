"""
This script tests the API deployed with ReaRAG/deploy/vllm_async_serving.py
"""
from transformers import AutoTokenizer
import re
from src_data.utils import get_response
import yaml

with open('src_data/test_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['vllm_async']

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
prompt = "What is 1+2?"

def extract_qwq_answer(response):
        response = response.split("**Final Answer**")[-1]
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, response)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            # Handle 'choose' mode
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
            return extracted_text
        else:
            return response.strip()

conversation = [
    {"role": "user", "content": prompt}
]
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

gen_params = {
            "max_tokens": 32768,
            "top_p": 0.85,
            "temperature": 0.99,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        }

response = get_response(prompt, config['api'], parameters=gen_params)
print(response)

print("*"*60)
extracted = extract_qwq_answer(response)
print(extracted)