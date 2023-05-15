import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

HOST = '127.0.0.1:5005'
URI = f'ws://{HOST}/api/v1/stream'


async def run():
    async with websockets.connect(URI, ping_interval=None) as websocket:
        prompt ="Once upon a time"
        await websocket.send(prompt)
        
        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['text']
                case 'stream_end':
                    return


async def print_response_stream():
    async for response in run():
        print(response, end='')
        sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.


if __name__ == '__main__':
    asyncio.run(print_response_stream())