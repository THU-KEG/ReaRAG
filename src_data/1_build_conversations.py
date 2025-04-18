import os
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # ThreadPool import
from transformers import AutoTokenizer
from typing import List, Dict, Any
import copy
import yaml

from src.rag_engine import RAGEngine
from src.prompts import data_construction_qwq, long_ans_prompt
from src_data.utils import read_jsonl, parse_reasoning_steps, format_thought_action, \
                    postprocess_codes, preprocess_question, get_response, \
                    load_completed_ids, save_jsonl_incremental, verify_reasoning_steps
random.seed(1)

with open('src_data/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['build_conv']

os.makedirs(config['output_dir'], exist_ok=True)
DOCUMENT_PROMPT = "Document {i}:\n{title}\n{document}"
ALLOWED_ACTIONS = ('search', 'finish')

def read_data():
    """
    Read the data from jsonl files
    Musique: 19938
    Hotpot: 7756
    NQ: 5000

    The data has these keys: dict_keys(['question', 'answer', 'paragraphs', 'prompt'])
    The key, `prompt` examples are as below:
    ------------------------ 
    Question:
    {question}

    Ground-truth answer:
    {gt_answer}

    Decompositions:
    {decompositions}

    Reasoning process with function call:
    ------------------------
    """
    data = []
    for files in config['input_files']:
        data.extend(read_jsonl(files))
    
    if config['n_sample'] > 0:
        data = data[:config['n_sample']]
    return data

def get_obs_from_env(query):
    query = preprocess_question(query)

    # Call rag_engine
    mem = rag_engine.Search(query)
    obs = rag_engine.Answer(query, long_ans_prompt, mem)

    return obs

def build_conversation(question: str) -> List[Dict[str, Any]]:
    prompt_init = f"Question: {question}\n\nReasoning process with function call:"
    conversation_qwq = [
        {"role": "system", "content": data_construction_qwq},
        {"role": "user", "content": prompt_init}
    ]
    conversation_wanted = [
        {"role": "system", "content": data_construction_qwq},
        {"role": "user", "content": prompt_init}
    ]

    retry_cnt = 0
    iter_cnt = 0

    while iter_cnt < config['iter_num_max']:
        # 0) Backup conversation
        conversation_qwq_backup = copy.deepcopy(conversation_qwq)
        conversation_wanted_backup = copy.deepcopy(conversation_wanted)

        # print(f"----Conversation----: \n{conversation_qwq}\n")
        # 1) Build prompt with special token from conversation
        prompt = llm_tokenizer.apply_chat_template(
            conversation_qwq,
            tokenize=False,
            add_generation_prompt=True
        )
        gen_params = {
            "max_tokens": 256,
            "top_p": 0.85,
            "temperature": 0.99,
            "skip_special_tokens": True,
            # "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        }

        try:
            # 2) Get model response
            # print(f"----Prompt----: \n{prompt}\n")
            response = get_response(prompt, llm_api, parameters=gen_params)
            print(f"----Response----: \n{response}\n")
        except Exception as e:
            print(f"Error in get_reponse: {e}")
            if retry_cnt < config['max_tries']:
                    conversation_qwq = conversation_qwq_backup
                    conversation_wanted = conversation_wanted_backup
                    retry_cnt += 1
                    continue
            else:
                return None

        # 3) Parse response and check if reasoning steps are valid
        reasoning_steps = parse_reasoning_steps(response)
        reasoning_steps = reasoning_steps[:1] # Only take the first reasoning step
        is_valid = verify_reasoning_steps(reasoning_steps, allowed_actions=ALLOWED_ACTIONS)
        if not is_valid:
            if retry_cnt < config['max_tries']:
                conversation_qwq = conversation_qwq_backup
                conversation_wanted = conversation_wanted_backup
                retry_cnt += 1
                continue
            else:
                return None

        # 4) Extract thought, action from response
        try:
            thoughts, actions, step_ids = postprocess_codes(reasoning_steps)
            thought, action, turn = thoughts[0], actions[0], step_ids[0]

            thought_action = format_thought_action(thought, action, turn)
            conversation_qwq += [{'role': 'assistant', 'content': thought_action}]
            conversation_wanted += [{'role': 'assistant', 'reasoning': thought_action}]

            action_type = action['function']
        except Exception as e:
            print(f"Error in postprocess_codes: {e}")
            if retry_cnt < config['max_tries']:
                    conversation_qwq = conversation_qwq_backup
                    conversation_wanted = conversation_wanted_backup
                    retry_cnt += 1
                    continue
            else:
                return None

        # 5) Handle "finish" or "search"
        if 'finish' in action_type:
            return conversation_wanted[1:] # Do not include the first system prompt
        
        elif 'search' in action_type:
            try:
                # Get observation, then append
                obs = get_obs_from_env(action['parameters']['query'])
                obs_qwq = f"Observation {turn}: {obs}"
            except Exception as e:
                print(f"Error in get_obs_from_env: {e}")
                if retry_cnt < config['max_tries']:
                    conversation_qwq = conversation_qwq_backup
                    conversation_wanted = conversation_wanted_backup
                    retry_cnt += 1
                    continue
                else:
                    return None
            conversation_qwq.append({'role': 'assistant', 'content': obs_qwq})
            conversation_wanted.append({'role': 'observation', 'content': obs})

        iter_cnt += 1

    return None # This might be bad data

    
def build(item):
    # Build reflection conversations
    conversation = build_conversation(item['question'])
    
    result = {
        "question": item['question'],
        "answer": item['answer'],
        "conversation": conversation,
        "id": item['id']
    }
    return result

def initialize():
    global llm_api, llm_tokenizer, rag_engine

    llm_api = config['llm_api']
    llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_tokenizer_path'], trust_remote_code=True)
    ans_tokenizer = AutoTokenizer.from_pretrained(config['ans_tokenizer_path'], trust_remote_code=True)

    # Init RAG engine
    rag_engine = RAGEngine(
        retriever_api = config['retriever_api'],
        generation_api = config['gen_api'],
        rag_config = {
            'top_k': config['top_k'],
        },
        generation_config = {
            'max_tokens': 1024,
            'top_p': 0.7,
            'temperature': 0.95,
            # 'stop': ["<|user|>", "<|endoftext|>", "<|assistant|>"]
        },
        tokenizer = ans_tokenizer      
    )

def main(data):
    initialize()

    for idx, item in enumerate(data):
        item['id'] = str(idx)

    # We'll submit tasks one by one, then collect + save as they finish
    futures = []
    reflection_data_file = os.path.join(config['output_dir'], config['output_file'])  # the partial-saves file

    # Figure out which items are already completed, and filter out processed items
    completed_ids = load_completed_ids(reflection_data_file)
    conv_data_to_process = [item for item in data 
                            if item["id"] not in completed_ids]

    # Now we build reflection conversation
    with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
        for item in conv_data_to_process:
            futures.append(executor.submit(build, item))

        # Use as_completed to get partial results immediately when a future is done
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Building reflection conversations")
        batch_buffer = []
        batch_size = config['save_batch_size']  # adjust as needed

        for future in pbar:
            result = future.result()
            batch_buffer.append(result)

            # If we have enough results, dump them to disk in JSONL format 
            # and clear the buffer
            if len(batch_buffer) >= batch_size:
                save_jsonl_incremental(reflection_data_file, batch_buffer)
                batch_buffer = []

        # After the loop, save any leftover items still in the buffer
        if batch_buffer:
            save_jsonl_incremental(reflection_data_file, batch_buffer)

if __name__ == "__main__":
    data = read_data()
    main(data)
