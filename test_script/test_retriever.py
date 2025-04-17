from functools import wraps
import time
import requests
import yaml

with open('src_data/test_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['retriever']

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        print(f"Retry {func.__name__} failed after {max} times")
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

@retry(max=10, sleep=1)
def _search(query: str, num: int = None, return_score=False):
    if num is None:
        num = self.topk
    url = f"{remote_url}/search"
    response = requests.post(url, json={"query": query, "top_n": num, "return_score": return_score})
    return response.json()

remote_url = config['api']
search_content = "Who is part of The Bruce Lee Band?"
search_result = _search(search_content, num=3)
print(search_result)
print(len(search_result))
print(search_result[0].keys())