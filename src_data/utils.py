import json
import re
import requests
from termcolor import colored

def get_response(prompt, base_url, parameters=None):
    if parameters is None:
        parameters = {
                "max_tokens": 1024,
                "top_p": 0.7,
                "temperature": 0.95,
                # "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            }
    try:
        rep = requests.post(
            base_url,
            json={
                'inputs': prompt,
                "parameters": parameters
            },
            headers={
                'Content-Type': 'application/json'
            },
            timeout=500
        )
        rep.raise_for_status()  # <-- raises an HTTPError if status != 200
    except Exception as e:
        print(colored(f"In get_response, Error parsing JSON: {e}", 'red'))
        print(colored(f"In get_response, HTTP status code: {rep.status_code}", 'red'))
        print(colored(f"In get_response, Raw response text: {rep.text}", 'red'))
        return ''
    
    # If everything is OK:
    response = rep.json()
    assert len(response) == 1
    generation = response['outputs'][0]['generated_text']
    return generation

def format_thought_action(thought, action, turn):
    return f"""Thought {turn}: {thought}
Action {turn}: 
```
{action}
```"""

def extract_code(text: str) -> str:
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    single_match = re.search(r'`([^`]*)`', text, re.DOTALL)
    if triple_match:
        return triple_match.group(1)
    elif single_match:
        return single_match.group(1)
    return text

def parse_reasoning_steps(text: str, pattern=None):
    """
    Parse a string containing Thought/Action/Observation steps (including multi-line)
    and return a list of dictionaries of the form:
    
    [
        {
            "1": {
                "Thought": "...",
                "Action": "...",  # Only content inside backticks (if present)
                "Observation": "..."
            }
        },
        {
            "2": {
                "Thought": "...",
                "Action": "...",
                "Observation": "..."
            }
        },
        ...
    ]
    """
    # Regex pattern to match lines that start with "Thought X:", "Action X:", or "Observation X:".
    if pattern is None:
        pattern = re.compile(r'^(Thought|Action|Observation)\s+(\d+):', re.MULTILINE)

    # This dictionary will accumulate:
    # data_dict[step_number] = {"Thought": ..., "Action": ..., "Observation": ...}
    data_dict = {}

    # We'll track the current label (Thought/Action/Observation) and step number
    current_label = None
    current_step = None
    last_pos = 0

    # Find all pattern occurrences in the text
    matches = list(pattern.finditer(text))

    for match in matches:
        # If we already have a label in progress, we can record its content
        if current_label is not None:
            # Slice the text from the last match's end to the start of this new match
            content = text[last_pos:match.start()].strip()
            # Store that content in data_dict
            data_dict[current_step][current_label] = content

        # Extract the new label and step
        label = match.group(1)       # "Thought", "Action", or "Observation"
        step = match.group(2)        # e.g. "1", "2", "3"

        # Ensure a dict for this step
        if step not in data_dict:
            data_dict[step] = {"Thought": None, "Action": None, "Observation": None}

        # Update current label/step
        current_label = label
        current_step = step
        # We'll slice from here next time
        last_pos = match.end()

    # Handle the final block after the last match
    if current_label is not None:
        content = text[last_pos:].strip()
        data_dict[current_step][current_label] = content

    # Post-process:
    #  - For each step, extract only the text inside triple backticks for "Action".
    for step_number in data_dict:
        action_text = data_dict[step_number]["Action"]
        if action_text:
            # Extract content inside triple backticks
            data_dict[step_number]["Action"] = extract_code(action_text)

    # Convert our dictionary to the desired list-of-dicts structure
    structured_data = []
    for step_number in sorted(data_dict.keys(), key=lambda x: int(x)):
        structured_data.append({step_number: data_dict[step_number]})

    return structured_data

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w') as f_out:
        for item in data:
            f_out.write(json.dumps(item) + '\n')
        
    print(f"Data saved at {filepath}")

def save_json(data, filepath):
    with open(filepath, 'w') as f_out:
        json.dump(data, f_out, ensure_ascii = False, indent=4)
        
    print(f"Data saved at {filepath}")
   
def preprocess_question(question):
    if "'" in question and '"' in question:
        question = question.replace("'", "\\'").replace('"', '\\"')
    return question

def verify_reasoning_steps(reasoning_steps, allowed_actions):
    n_steps = len(reasoning_steps)
    for turn, step in enumerate(reasoning_steps):
        for reasoning_iter, reasoning in step.items():

            # thought = reasoning['Thought']
            action = reasoning['Action']
            # obs = step[str(step_number)]['Observation']
            
            # Verify if action is valid dict
            try:
                action_type = eval(action)['function']
            except Exception as e:
                print(f"Error in verify_reasoning_steps, not valid dict: {e}\nGot action: {action}")
                return False

            # Verify if action is allowed
            try:
                assert action_type in allowed_actions
            except Exception as e:
                print(f"Error in verify_reasoning_steps, actions not allowed: {e}\nGot action: {action}")
                return False

            # if step_number < n_steps:
            #     if thought is None or action is None or obs is None:
            #         return False
            
            # # if step_number is last steps
            # if step_number == n_steps:
            #     if thought is None or action is None:
            #         return False
    return True

def postprocess_codes(reasoning_steps):
    """
    Extract Thought and Action, then extract the dict from Action.

    Return:
    - thought: str
    - action: dict
    - is_valid: bool
    """

    step_ids = []
    thoughts = []
    actions = []
    for steps in reasoning_steps:
        for step_idx, step in steps.items():
            # thought = f"Thought {step_idx}: {step['Thought']}"
            action = eval(extract_code(step['Action']))

            thoughts.append(step['Thought'])
            actions.append(action)
            step_ids.append(step_idx)

    return thoughts, actions, step_ids

def save_jsonl_incremental(filename, data_list):
    """Append data to a JSONL file incrementally."""
    with open(filename, "a") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def load_completed_ids(filename):
    """Load all od from your JSONL file and return as a set."""
    import json
    completed = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                completed.add(data["id"])
                if data['conversation'] is not None:
                    completed.add(data["id"])
    except FileNotFoundError:
        pass
    return completed