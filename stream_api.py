import torch
from transformers import AutoTokenizer, StoppingCriteriaList
from threading import Thread
import gc
import traceback
import asyncio
import json
from websockets.server import serve

from inference_utils import load_quant, Stream, Iteratorize, _SentinelTokenStoppingCriteria

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update to whichever GPTQ model you want to load
MODEL_NAME = './models/TheBloke/stable-vicuna-13B-GPTQ'
# Update the model weight that you want to load for inference.
MODEL_PATH = './models/TheBloke/stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors'


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


def generate_with_callback(callback=None, **kwargs):
    kwargs['stopping_criteria'].append(Stream(callback_func=callback))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.generate(**kwargs)


def generate_with_streaming(**kwargs):
    return Iteratorize(generate_with_callback, kwargs, callback=None)


PATH = '/api/v1/stream'


async def _handle_connection(websocket, path):
    if path != PATH:
        print(f'Streaming api: unknown path: {path}')
        return

    async for message in websocket:
        # Use plain text for now, can change to JSON string.
        input_text = message

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt").to(DEV)

        # handle stopping strings
        stopping_strings = ['Human:']
        stopping_criteria_list = StoppingCriteriaList()
        sentinel_token_ids = [tokenizer.encode(
            string, add_special_tokens=False, return_tensors='pt').to(DEV) for string in stopping_strings]
        starting_idx = len(input_ids[0])
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(
            sentinel_token_ids, starting_idx))

        # hardcode generation parameters
        generate_params = {
            'input_ids': input_ids,
            'max_length': 100,
            'temperature': 1.0,
            'do_sample': True,
            "top_p": 0.9,
            'stopping_criteria': stopping_criteria_list,
        }

        # As we stream, only send the new bytes.
        skip_index = 0
        message_num = 0

        # Generate tokens one by one
        with Iteratorize(generate_with_callback, generate_params, callback=None) as generator:
            for output in generator:
                # Decode the entire generated text so far
                generated_text = tokenizer.decode(output.cpu())
                # Only send the new part of the text
                to_send = generated_text[skip_index:]
                # remove bos token
                if not skip_index:
                    to_send = to_send.replace(tokenizer.bos_token, "")
                    to_send = to_send.strip()
                    
                await websocket.send(json.dumps({
                    'event': 'text_stream',
                    'message_num': message_num,
                    'text': to_send
                }))

                await asyncio.sleep(0)

                skip_index += len(to_send)
                message_num += 1

        await websocket.send(json.dumps({
            'event': 'stream_end',
            'message_num': message_num
        }))


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(port: int):
    address = '0.0.0.0'  # Listen on all addresses

    print(f'Starting streaming server at ws://{address}:{port}')

    asyncio.run(_run(host=address, port=port))


if __name__ == '__main__':
    _run_server(5005)
