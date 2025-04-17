rearag_system_prompt = '''Your task is to solve a question answering task. To improve your solving accuracy, please conduct reasoning process interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action are in the form of function, there are two types:

# Available functions:
(1) search
{
    "name": "search",
    "description": "It can help you find useful information through the internet or local knowledge base. You can use this tool to access external knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "description": "what you want to search"
            }
        },
        "required": [
            "query"
        ]
    }
}

(2) finish
{
    "name": "finish",
    "description": "You can use this function to make a conclusion from the reasoning process and give the final answer. The reasoining process is completed after this `finish` function is called",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "description": "the final answer"
            }
        },
        "required": [
            "answer"
        ]
    }
}

Please follow the format strictly.
'''

# Define some templates for Answer() in rag_engine 
short_ans_prompt = 'Context: {}\n\nAnswer the Question based on the given Contexts. Only give me the answer and do not output any other words.\n\nQuestion: {}\n\nAnswer:'
long_ans_prompt = 'If the Question is comparison type, do not refer the given Context, else, answer the Question based on the given Contexts.\n\nContext: {}\n\nQuestion: {}\n\nAnswer:'


# Prompt to extract short answer
extract_short_ans_prompt = """The Reference Answer is the final answer to the question. It's the final deterministic answer, your task is to give concise version of it. Only give me the short answer and do not output any other words.
[Question]
{question}
[Reference answer]
{reference_ans}

Only give me the short answer and do not output any other words. For yes, or no answer, only answer it short. Give the shortest answer possible.
"""

data_construction_qwq = '''Your task is to solve a question answering task. To improve your solving accuracy, please conduct reasoning process following this sequence: Thought, Action, Observation steps. Thought can reason about the current situation, and Action are in the form of function, there are two types:

# Available functions:
(1) search
{
    "name": "search",
    "description": "It can help you find useful information through the internet or local knowledge base. You can use this tool to access external knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "description": "what you want to search"
            }
        },
        "required": [
            "query"
        ]
    }
}

(2) finish
{
    "name": "finish",
    "description": "You can use this function to make a conclusion from the reasoning process and give the final answer. The reasoining process is completed after this `finish` function is called",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "description": "the final answer"
            }
        },
        "required": [
            "answer"
        ]
    }
}

# Some important rules you must follow:
(1) Please follow the function calling format above strictly. 
(2) A set of `Thought`, `Action`, and `Observation` is considered as one reasoning step. Add numbering after each `Thought`, `Action`, and `Observation` to indicate the sequence of the reasoning steps.
(3) Please give your `Thought` first, then the `Action`, and finally the `Observation`, follow the format as shown in the in-context examples below.
(3) In your `Thought`, you should perform reflection when necessary to ensure the correctness of your reasoning process, such as: "Wait! Maybe I made some mistakes! I need to rethink from scratch", "Alternatively, we can...", "Hold on, let's try another approach", etc. 
(4) Give your `Action` in the form of function call, as shown in in-context examples below.
(5) You should not provide information based on your own knowledge, only use the information provided in the context.

# Some example of reflection text:
""""There is no enough information from the previous steps. I need to plan my query again.",
"No context found from observation. Let me restart the reasoning process.",
"Missing information. Let me restructure my query.",
"Wait! Maybe I made some mistakes! I need to rethink from scratch.",
"I think I need to take a step back and reconsider my approach.",
"I need to reevaluate my reasoning process. Let's start over.",
"I need to reflect on my reasoning. Let's try a different approach.",
"I need to reconsider my reasoning. Let's take a step back.",
"Hold on! I should pause and confirm whether my previous logic holds up.",
"Wait a second! I might need to break this down into smaller, more manageable parts.",
"I’m going to re-examine the problem statement to ensure I'm answering the right question.",
"Wait, let me carefully verify each step of the process.",
"I should revisit the assumptions made in earlier steps and verify their validity.""""

# In-context examples:
Question:
What was the wettest year in the second largest city in the state where Yuma's Library District is located?

Reasoning process with function call:
Thought 1: First, we need to determine where Yuma's Library District is located. We can use the 'search' function.
Action 1:
```
{'function': 'search', 'parameters': {'query': 'Where is Yuma's Library District?'}}
```
Observation 1: The Yuma's Library District is located in Yuma.

Thought 2: Wait, perhaps we need to find the administrative territorial entity of Yuma.
Action 2:
```
{'function': 'search', 'parameters': {'query': 'Yuma located in which administrative territorial entity?'}}
```
Observation 2: The City of Yuma is a Home Rule Municipality that is the most populous city in Yuma County.

Thought 3: Next, we need to find out Yuma County Library District is located in which state.
Action 3:
```
{'function': 'search', 'parameters': {'query': 'Which state is Yuma County Library District located?'}}
```
Observation 3: The answer is Arizona.

Thought 4: emm, What is the second largest city in Arizona ?
Action 4:
```
{'function': 'search', 'parameters': {'query': 'What is the second largest city in Arizona'}}
```
Observation 4: It should be Tucson.

Thought 5: we need to Tucson's wettest year. Use the 'search' function:
Action 5:
```
{'function': 'search', 'parameters': {'query': 'What was Tucson's wettest year?'}}
```
Observation 5: The answer is 1905.

Thought 6: Based on the decomposition steps, we have determined the following:

(1) Yuma's Library District is in Yuma County.
(2) Yuma County is located in Arizona.
(3) The second largest city in Arizona is Tucson.
(4) Tucson's wettest year is 1905.
Action 6:
```
{'function': 'finish', 'parameters': {'answer': '1905'}}
```
'''