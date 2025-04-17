"""
This script clean the conversations build from build_conversations.py, the criteria for cleaning is:
* the conversation must be valid, and the answer achieved F1 score > 0.
"""

import re
import copy
import os
import argparse
import yaml

from src_data.utils import parse_reasoning_steps, read_jsonl, save_json
from src_data.metrics import qa_f1_score

with open('src_data/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['clean_conv']

def check_valid_conversation(convs, gt_answer):
    """
    The rules for a valid conversation are as follows:
    (1) First conv must be content from 'user'
    (2) If the conv is from 'assistant', and its 'action' is not 'reflect', this conv must be followed by 'observation'
    (4) The numbering of thought must be in order
    (5) The last conv must be from 'assistant' and its 'action' is 'finish'
    (6) The prediction must score > 0 in F1 score
    """

    # Helper function to extract the (thought_number, action) from an assistant entry
    def get_assistant_reasoning(assistant_conv):
        """
        Expected structure of 'assistant_conv["reasoning"]' is something like:
        {
            '2': {
                'Thought': 'some thought text',
                'Action': '{...}'
            }
        }
        We want to extract:
          - The numeric key (here, '2')
          - The 'Action' string inside the dictionary
        """
        reasoning_dict = assistant_conv.get("reasoning", {})
        if not reasoning_dict:
            return None, None

        # Typically there's only one key, e.g. '1', '2', etc.
        # We'll retrieve that single key (if it exists).
        thought_num_str = list(reasoning_dict.keys())[0]
        action_str = reasoning_dict[thought_num_str].get("Action", "")
        return thought_num_str, action_str

    # 1) Check that the first conversation entry is from the user
    if not convs:
        return False, "Conversation is empty."
    if convs[0]["role"] != "user":
        return False, "First conversation must be from user."

    # Variables to keep track of the previous assistant's action for rule #2 and #3
    prev_assistant_action = None

    # For rule (4): numbering of thoughts in ascending order
    expected_thought_num = 1

    # Iterate over the conversation entries
    for i, conv in enumerate(convs):
        role = conv["role"]

        if role == "assistant":
            # Extract thought number and action
            thought_str, action_str = get_assistant_reasoning(conv)
            if not thought_str or not action_str:
                return False, f"Assistant entry at index {i} missing reasoning or action."

            # (4) Check numbering of thought must be in ascending order
            try:
                thought_num = int(thought_str)
            except ValueError:
                return False, f"Thought number '{thought_str}' is not an integer."

            if thought_num != expected_thought_num:
                return False, (
                    f"Thought number out of order. Expected {expected_thought_num}, "
                    f"got {thought_num} at index {i}."
                )
            expected_thought_num += 1

            # Check rule (2) and (3) regarding what must follow
            prev_assistant_action = eval(action_str)['function']

            # If the previous assistant action was NOT "reflect", we needed an observation here (rule #2).
            if i + 1 < len(convs) and convs[i + 1]["role"] != "observation":
                return False, (
                    f"Expected 'observation' after assistant action, but found "
                    f"'{convs[i + 1]['role']}' at index {i + 1}."
                )

        elif role == "observation":
            
            if "search" != prev_assistant_action:
                return False, (
                    f"Expected 'search' action before 'observation' action, but found "
                    f"'{prev_assistant_action}' at index {i}."
                )
                
            # # Clear the prev_assistant_action after observation
            prev_assistant_action = None

        elif role == "user":
            # If the previous assistant action was "reflect", we needed a user message (rule #3).
            # That means it's valid to have user now if the prev action was reflect.
            if i != 0:
                return False, (
                    f"Expected 'observation' after an assistant action (not reflect), "
                    f"but found 'user' at index {i}."
                )
            # Clear the prev_assistant_action after user
            prev_assistant_action = None

        else:
            return False, f"Unknown role '{role}' at index {i}."

    # (5) The last conv must be from 'assistant' and its 'action' is 'finish'
    last_conv = convs[-1]
    if last_conv["role"] != "assistant":
        return False, "Last conversation entry must be from 'assistant'."
    # Extract the last assistant's action
    _, last_action_str = get_assistant_reasoning(last_conv)
    if "finish" not in last_action_str:
        return False, "Last assistant action must be 'finish'."

    # If all rules are satisfied, now we check if prediction is correct (rule #6)
    prediction = eval(last_action_str)['parameters']['answer']
    score = qa_f1_score(gt_answer, prediction)
    if score == 0:
        return False, "Prediction is incorrect."

    # Finally, return True
    return True, "Conversation is valid."

def main():
    data = read_jsonl(os.path.join(config['input_dir'], config['input_file']))
    file_path_validity = os.path.join(config['output_dir'], config['output_file_validity'])
    clean_path = os.path.join(config['output_dir'], config['output_file'])

    print(f"Number of data: {len(data)}")

    data_validity = []
    for d in data:
        item = d['conversation']
        if item: # some data is empty
            ori_conv = copy.deepcopy(item)

            # Parse the data
            tgt_data_processed = []
            for i, conv in enumerate(item):
                if conv['role'] == 'assistant':
                    # Try to separate reasoning steps that contains multiple Thought if they are combined
                    pattern = re.compile(r'^(Thought|Action)\s+(\d+):', re.MULTILINE)
                    reasoning_steps = parse_reasoning_steps(conv['reasoning'], pattern)
                    tgt_data_processed.extend([{'role': 'assistant', 'reasoning': step} for step in reasoning_steps])
                else:
                    tgt_data_processed.append(conv)

            # Check if the conversation is valid
            result, message = check_valid_conversation(tgt_data_processed, d['answer'])

            data_validity.append({
                'conversation': ori_conv,
                'valid': result,
                'valid_message': message
                }
            )
        else:
            data_validity.append({
                'conversation': None,
                'valid': False,
                'valid_message': "Empty conversation"
                }
            )

    # Save the processed data
    save_json(data_validity, file_path_validity)

    # Filter out invalid conversations
    data = [item['conversation'] for item in data_validity if item['valid']]
    print(f"Number of valid conversations: {len(data)}")
    print(f"Number of invalid conversations: {len(data_validity) - len(data)}")
    save_json(data, clean_path)


if __name__ == "__main__":
    main()