
import requests
from termcolor import colored

class RAGEngineBase:
    def __init__(self, retriever_api, generation_api, rag_config, generation_config, tokenizer):
        """
        Base class for a Retrieval-Augmented Generation (RAG) engine.
        Parameters:
        - retriever_api: API or interface for retrieval.
        - generation_api: API or interface for text generation.
        - rag_config: Configuration for the retrieval component.
        - generation_config: Configuration for the generation component.
        - tokenizer: Tokenizer for answer generation.
        """
        self.retriever_api = retriever_api
        self.generation_api = generation_api
        self.rag_config = rag_config
        self.generation_config = generation_config
        self.tokenizer = tokenizer

    def Search(self):
        """
        Abstract method for implementing the search functionality.
        Must be overridden in a derived class.
        """
        raise NotImplementedError("The 'Search' method must be implemented in a subclass.")

    def Answer(self):
        """
        Abstract method for implementing the answer generation functionality.
        Must be overridden in a derived class.
        """
        raise NotImplementedError("The 'Answer' method must be implemented in a subclass.")

    def get_response(self, prompt, base_url, parameters=None):
        if parameters is None:
            parameters = {
                    "max_tokens": 1024,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "stop": None,
                    "skip_special_tokens": False
                }
        try:
            rep = requests.post(
                base_url,
                json={
                    'inputs': prompt,
                    'stream': False,
                    "parameters": parameters
                },
                headers={
                    'Content-Type': 'application/json'
                },
                timeout=360
            )
            rep.raise_for_status()  # <-- raises an HTTPError if status != 200
            
        except Exception as e:
            print(colored(f"In get_response, Error parsing JSON: {e}", 'red'))
            print(colored(f"In get_response, HTTP status code: {rep.status_code}", 'red'))
            print(colored(f"In get_response, Raw response text: {rep.text}", 'red'))
            raise

        # If everything is OK:
        response = rep.json()
        assert len(response) == 1
        generation = response['outputs'][0]['generated_text']
        return generation

class RAGEngine(RAGEngineBase):
    def __init__(self, retriever_api, generation_api, rag_config, generation_config, tokenizer):
        super().__init__(retriever_api, generation_api, rag_config, generation_config, tokenizer)

    def Search(self, query):
        try:
            rep = requests.post(
                url=self.retriever_api,
                json={
                    'query': query,
                    'top_n': self.rag_config['top_k'],
                    "return_score": False
                },
                headers={
                    'Content-Type': 'application/json'
                },
                timeout=300
            )
            rep.raise_for_status()  # <-- raises an HTTPError if status != 200

        except Exception as e:
            print(colored(f"In Search, Error in HTTP request: {e}", 'red'))
            raise
        
        # If everything is OK:
        search_result = rep.json()
        return search_result 
    
    def Answer(self, question, prompt_template, memory=None, summary_chain=None):
        chunks = []
        for _ in memory:
            if _ not in chunks:
                chunks.append(_['contents'])
        
        if summary_chain is not None:
            for _ in summary_chain:
                if _ not in chunks:
                    chunks.append(_)
        prompt = prompt_template.format('\n\n'.join(chunks), question)

        conversation_chain = [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": ""},  # Placeholder for model generation
        ]

        prompt = self.tokenizer.apply_chat_template(
                    conversation_chain,
                    add_special_tokens=False,  # Set True if you want special tokens
                    tokenize=False,  # Set True if you want tokenized result
                    add_generation_prompt=True  # Ensures correct format for model to continue generating
                )
        try:
            response = self.get_response(prompt, self.generation_api, self.generation_config)
        except Exception as e:
            print(colored(f"In Answer, Error in get_response: {e}", 'red'))
            raise
        return response