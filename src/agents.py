import requests
from typing import List, Dict, Any, Union
import copy
from termcolor import colored
from src.prompts import rearag_system_prompt, short_ans_prompt, long_ans_prompt, extract_short_ans_prompt
from src.utils import print_code

class ReaRAGAgent():
    def __init__(self, agent_api, tokenizer, allowed_actions, rag_engine, 
                    iter_num_max, retry_max, agent_config, agent_utils):
        self.agent_api = agent_api
        self.tokenizer = tokenizer
        self.allowed_actions = allowed_actions
        self.rag_engine = rag_engine
        self.iter_num_max = iter_num_max
        self.retry_max = retry_max
        self.agent_config = agent_config
        self.agent_utils = agent_utils

    def init_agent(self, question):
        """
        Initialize conversation chain, reasoning chain, and summary_chain
        """
        self.question = question

        self.conversation_chain = [
            {"role": "system", "content": rearag_system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": ''},
        ]
        self.reasoning_chain = [] # dict type: thought, action, observation
        self.summary_chain = [] # str type:(question, answer)
        self.cur_iter_num = 1

    def run(self):
        """
        Infer single data: Given a question, interact with environment, then return the final answer
        """
        final_answer = None
        retry_cnt = 0
        while self.cur_iter_num <= self.iter_num_max:
            conv_backup = copy.deepcopy(self.conversation_chain)
            summary_backup = copy.deepcopy(self.summary_chain)
            reasoning_backup = copy.deepcopy(self.reasoning_chain)

            # prompt = self.tokenizer.decode(self.agent_utils.tokenize_data(self.conversation_chain, self.system_prompt)['text'])
            prompt = self.tokenizer.apply_chat_template(
                self.conversation_chain,
                add_special_tokens=False,
                tokenize=False,  # Set True if you want tokenized result
                add_generation_prompt=True  # Ensures correct format for model to continue generating
            )
            
            status, status_message =  self.step(prompt)

            if status == "repeat":
                if retry_cnt < self.retry_max:
                    # Reset conversation_chain, summary_chain, reasoning_chain
                    self.conversation_chain = copy.deepcopy(conv_backup) 
                    self.summary_chain = copy.deepcopy(summary_backup)
                    self.reasoning_chain = copy.deepcopy(reasoning_backup)
                    retry_cnt += 1
                    continue
                else:
                    break

            elif status == "continue":
                self.cur_iter_num += 1
        
            elif status == "finish":
                final_answer = status_message
                break
            
            # print(f"Iteration {self.cur_iter_num}: {status}")
            # print_code(self.reasoning_chain)

        return final_answer
        
    def step(self, prompt):
        """
        Perform one step to prompt the model one time.
        Next perform specific action based on the response, and return the result of action if exist
        """

        # (1) Get llm agent response
        try:
            agent_response = self.get_agent_response(prompt, self.agent_api)
        except Exception as e:
            print(f"Error in get_agent_response: {e}")
            return "repeat", "Error in get_agent_response"

        # (2) Extract content from the response (Thought, action)
        try:
            thoughts, actions = self.agent_utils.postprocess_agent_response(agent_response)
        except Exception as e:
            print(f"Error in postprocess_agent_response: {e}")
            return "repeat", "Error in postprocess_agent_response"

        # (4) Perform the action
        for thought, action in zip(thoughts, actions):
            # (4.1) Check if the action is allowed
            action_type = action['function'] # Action is in the form of dict, {'function': '...', 'parameters': {'query': '...'}}

            if action_type not in self.allowed_actions:
                return "repeat", "Action not allowed"
            
            # (4.2) Perform the action
            if action_type == "search":
                try:
                    self.handle_search_step(
                        thought, action, agent_response
                    )
                except Exception as e:
                    print(f"Error in handle_search_step: {e}")
                    return "repeat", "Error in handle_search_step"

            elif action_type == "finish":
                try:
                    reference_ans = action['parameters']['answer']
                    final_answer = self.handle_finish_step(reference_ans)
                except Exception as e:
                    print(f"Error in handle_finish_step: {e}")
                    return "repeat", "Error in handle_finish_step"
                
                # Store final state in the reasoning chain
                self.reasoning_chain.append({
                    'thought': thought,
                    'action': action,
                    'observation': final_answer
                })
                return "finish", final_answer

            elif action_type == "reflect":
                self.reasoning_chain.append({
                    'thought': thought,
                    'action': action,
                    'observation': None
                })

        return "continue", None

        
    def handle_search_step(
        self,
        thought: str,
        action: Dict[str, Any],
        agent_response: str,
    ) -> None:
        """
        Handle a 'search' action within the RAG loop.
        Update conversation, summary_chain, reasoning_chain in-place.
        """
        query = self.agent_utils.preprocess_query(action['parameters']['query'])

        # Call rag_engine
        mem = self.rag_engine.Search(query)
        observation = self.rag_engine.Answer(query, long_ans_prompt, mem) # LongAnswer

        # Store summary and reasoning chain
        self.summary_chain.append(f"{query}\n{observation}")
        self.reasoning_chain.append({
            'thought': thought,
            'action': action,
            'observation': observation
        })

        # Update conversation
        self.conversation_chain[-1] = {"role": "assistant", "content": agent_response}
        self.conversation_chain.append({"role": "observation", "content": observation})
        self.conversation_chain.append({"role": "assistant", "content": ''})

    def handle_finish_step(
        self, reference_ans: str
    ) -> str:
        """
        Handle a 'finish' action in the RAG loop.
        Returns the final short answer.
        """
        # # Optionally: get global memory again for final short answer
        # mem = self.rag_engine.Search(self.question)
        # # final_answer = self.rag_engine.Answer(self.question, short_ans_prompt, self.summary_chain + mem)
        # final_answer = self.rag_engine.Answer(self.question, short_ans_prompt, mem, self.summary_chain)

        prompt = extract_short_ans_prompt.format(question=self.question, reference_ans=reference_ans)
        conversation_chain = [
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
                    conversation_chain,
                    add_special_tokens=False,  # Set True if you want special tokens
                    tokenize=False,  # Set True if you want tokenized result
                    add_generation_prompt=True  # Ensures correct format for model to continue generating
                )
        final_answer = self.rag_engine.get_response(prompt, self.rag_engine.generation_api, self.rag_engine.generation_config)

        return final_answer
    
    def get_agent_response(self, prompt, base_url):
        if self.agent_config["truncate"]:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens)> self.agent_config["model_max_length"]:
                print(f"Warning: Current prompt length: {len(tokens)},  exceeds model_max_length: {self.agent_config['model_max_length']}, truncating...")
                prompt = self.agent_utils.truncate(self.tokenizer, tokens, self.agent_config["model_max_length"])

        try:
            rep = requests.post(
                base_url,
                json={
                    'inputs': prompt,
                    'stream': False,
                    "parameters": {
                        "max_tokens": self.agent_config["max_tokens"],
                        "top_p": self.agent_config["top_p"],
                        "temperature": self.agent_config["temperature"],
                        "stop": self.agent_config["stop"],
                        "skip_special_tokens": False
                    }
                },
                headers={
                    'Content-Type': 'application/json'
                },
                timeout=360
            )
            rep.raise_for_status()  # <-- raises an HTTPError if status != 200

        except Exception as e:
            print(f"Error in get_agent_response: {e}")
            print(rep.text)
            print("-----")
            print(rep.json())
            raise
        
        # If everything is OK:
        response = rep.json()
        assert len(response) == 1
        generation = response['outputs'][0]['generated_text']
        return generation