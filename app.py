from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, StoppingCriteriaList
from inference_utils import load_quant, _SentinelTokenStoppingCriteria

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = './models/mzbac/stable-vicuna-13B-GPTQ' # Update to whichever GPTQ model you want to load
MODEL_PATH = './models/mzbac/stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors' # Update the model weight that you want to load for inference.

class SingletonModelTokenizer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonModelTokenizer, cls).__new__(cls)
            cls._instance.model = load_quant(MODEL_NAME, MODEL_PATH, 4, 128)
            cls._instance.model.to(DEV)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return cls._instance


singleton = SingletonModelTokenizer()
model = singleton.model
tokenizer = singleton.tokenizer

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json(force=True)
    text = data.get('text')
    min_length = data.get('min_length', 0)
    max_length = data.get('max_length', 50)
    top_p = data.get('top_p', 0.95)
    temperature = data.get('temperature', 0.6)
    stopping_strings = data.get('stopping_strings', [])

    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEV)

    # handle stopping strings
    stopping_criteria_list = StoppingCriteriaList()
    if len(stopping_strings)>0:
        sentinel_token_ids = [tokenizer.encode(
            string, add_special_tokens=False, return_tensors='pt').to(DEV) for string in stopping_strings]
        starting_idx = len(input_ids[0])
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(
            sentinel_token_ids, starting_idx))

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            stopping_criteria=stopping_criteria_list,
        )

    generated_text = tokenizer.decode([el.item() for el in generated_ids[0]])

    # Remove the BOS token from the generated text
    generated_text = generated_text.replace(tokenizer.bos_token, "")
    return jsonify({'generated_text': generated_text.strip()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
