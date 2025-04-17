import argparse
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify

app = Flask(__name__)

global llm_model

@app.post("/generate")
def generate():

    # Get JSON data from request
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON data'}), 400

    prompt = data.get('inputs', '')
    parameters = data.get('parameters', {})

    # Set up sampling parameters.
    sampling_params = SamplingParams(**parameters)

    # Generate text
    outputs = llm_model.generate(prompt, sampling_params)

    # Extract the generated text
    return_outputs = []
    for output in outputs:
        return_outputs.append({
            "prompt": output.prompt,
            "stop_reason": output.outputs[0].stop_reason,
            "generated_text": output.outputs[0].text
        })

    # Return the response in the expected format
    response = {'outputs': return_outputs}
    return jsonify(response)


def main(args):
    global llm_model
    
    # Load model and tokenizer
    llm_model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        seed=args.seed,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Start the Flask app
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # vllm
    parser.add_argument("--model_path", type=str, default="gpt2-medium")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)

    # flask
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to run the Flask app on")
    parser.add_argument("--port", type=int, default=9891, help="Port to run the Flask app on")

    args = parser.parse_args()

    main(args)
