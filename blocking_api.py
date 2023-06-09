import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import logging

import torch
from transformers import AutoTokenizer, StoppingCriteriaList
from inference_utils import load_quant, _SentinelTokenStoppingCriteria

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update to whichever GPTQ model you want to load
MODEL_NAME = './models/mzbac/stable-vicuna-13B-GPTQ'
# Update the model weight that you want to load for inference.
MODEL_PATH = './models/mzbac/stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors'

# Setup logging
logging.basicConfig(level=logging.INFO)


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


class GenerateHandler:
    @staticmethod
    def handle_request(handler, body):
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/json')
        handler.end_headers()

        text = body['prompt']
        min_length = body.get('min_length', 0)
        max_length = body.get('max_length', 1000)
        top_p = body.get('top_p', 0.95)
        top_k = body.get('top_k', 40)
        typical_p = body.get('typical_p', 1)
        do_sample = body.get('do_sample', True)
        temperature = body.get('temperature', 0.6)
        no_repeat_ngram_size = body.get('no_repeat_ngram_size', 0)
        num_beams = body.get('num_beams', 1)
        stopping_strings = body.get('stopping_strings', ['Human:', ])

        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEV)

        # handle stopping strings
        stopping_criteria_list = StoppingCriteriaList()
        if len(stopping_strings) > 0:
            sentinel_token_ids = [tokenizer.encode(
                string, add_special_tokens=False, return_tensors='pt').to(DEV) for string in stopping_strings]
            starting_idx = len(input_ids[0])
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(
                sentinel_token_ids, starting_idx))

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                min_length=min_length,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k,
                typical_p=typical_p,
                do_sample=do_sample,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                stopping_criteria=stopping_criteria_list,
            )

        generated_text = tokenizer.decode(
            [el.item() for el in generated_ids[0]], skip_special_tokens=True)

        response = json.dumps({'results': [{'text': generated_text}]})
        handler.wfile.write(response.encode('utf-8'))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))
        if self.path == '/api/v1/generate':
            GenerateHandler.handle_request(self, body)
        else:
            self.send_error(404)


def _run_server(port: int, share: bool = False):
    address = '0.0.0.0'
    server = ThreadingHTTPServer((address, port), Handler)
    logging.info('Server is running on http://{}:{}'.format(address, port))
    server.serve_forever()


def start_server(port: int, share: bool = False):
    Thread(target=_run_server, args=[port, share]).start()


if __name__ == '__main__':
    start_server(5000)
