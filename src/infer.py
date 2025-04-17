from src.rag_engine import RAGEngine
from src.agents import ReaRAGAgent
from src.utils import AgentUtils, print_code
from transformers import AutoTokenizer

QUESTION = "Where was the author of Hannibal and Scipio educated at?"
ALLOWED_ACTIONS = ["search", "finish"]
ITER_NUM_MAX = 15 # Number of inference iterations per samples, to avoid infinite loop
MAX_TRIES = 20 # Number of tries to get a valid generated code 

def main(args):
    # Load tokenizer
    rearag_tokenizer = AutoTokenizer.from_pretrained(args.rearag_tokenizer_path, trust_remote_code=True)
    ans_tokenizer = AutoTokenizer.from_pretrained(args.ans_tokenizer_path, trust_remote_code=True)

    # Init RAG engine
    rag_engine = RAGEngine(
        retriever_api = args.retriever_api,
        generation_api = args.gen_api,
        rag_config = {
            'top_k': args.top_k,
        },
        generation_config = {
            'max_tokens': 1024,
            'top_p': 0.7,
            'temperature': 0.95,
            'stop': ["<|user|>", "<|endoftext|>", "<|assistant|>"]
        },
        tokenizer = ans_tokenizer      
    )

    # Init agent config
    agent_config = {
        'truncate': True,
        'model_max_length': 8192 - 2 - 1024,
        'max_tokens': 1024,
        'temperature': 1.0,
        'top_p': 0.85,
        'stop': ["<|user|>", "<|observation|>", "<|assistant|>"]
    }

    # Init agent
    agent_utils = AgentUtils()
    llm_agent = ReaRAGAgent(
        agent_api = args.agent_api,
        tokenizer= rearag_tokenizer,
        allowed_actions = ALLOWED_ACTIONS,
        rag_engine = rag_engine, 
        iter_num_max = ITER_NUM_MAX,
        retry_max=MAX_TRIES,
        agent_config = agent_config,
        agent_utils = agent_utils
    )
    llm_agent.init_agent(QUESTION)

    # Run the agent, get answer
    final_answer = llm_agent.run()
    print_code(llm_agent.reasoning_chain)
    print(f"Final answer:\n{final_answer.strip()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ReaRAG inference')
    parser.add_argument('--agent_api', type=str, 
                        required=True, help="API to ReaRAG model")
    parser.add_argument('--retriever_api', type=str, 
                        required=True, help="search API")
    parser.add_argument('--gen_api', type=str, 
                        required=True, help="generation API for RAG engine")
    parser.add_argument('--rearag_tokenizer_path', type=str, 
                        required=True, help="Tokenizer path for ReaRAG")
    parser.add_argument('--ans_tokenizer_path', type=str, 
                        required=True, help="Tokenizer path for answer generation LLM in rag engine")
    parser.add_argument('--top_k', type=int, default=3,
                        help="number of retrieved documents")
    args = parser.parse_args()

    main(args)
