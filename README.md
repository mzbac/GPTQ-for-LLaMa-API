# GPTQ-for-LLaMa-API

Thanks to the awesome repository from qwopqwop200 on GitHub (https://github.com/qwopqwop200/GPTQ-for-LLaMa), we are now able to run large LLM models on consumer-grade GPUs. However, the original tool only provided a CLI-like inference. In this repository, it uses qwopqwop200's GPTQ-for-LLaMa implementation and serves the generated text via a simple Flask API.

## Hardware Requirements
An NVIDIA GPU with CUDA support is required for running the model. Please ensure that you have the correct NVIDIA drivers installed on your system and that CUDA is correctly set up. For details on the memory requirement of the GPU, please refer to https://github.com/qwopqwop200/GPTQ-for-LLaMa#result.

## Installation

```bash
conda create -n GPTQ python=3.10.9
conda activate GPTQ
git clone git@github.com:mzbac/GPTQ-for-LLaMa-API.git
cd GPTQ-for-LLaMa-API
pip install -r requirements.txt
```


## Downloading models

Run the Python script to download the model from Hugging Face. For example:
```
python download.py TheBloke/stable-vicuna-13B-GPTQ
```

## Selecting a GPU
By default, the script will use the first available GPU. If you have multiple GPUs and want to select which one to use, you can do so by setting the CUDA_VISIBLE_DEVICES environment variable before running your script.

For example, to use the first GPU, you can do:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Usage
1. Update the model name and model weight path in app.py and run.

```
python app.py
```
The server will start on localhost port 5000.

2. To generate text, send a POST request to the /generate endpoint. The request body should be a JSON object with the following keys:

- text: The input text (required).
- min_length: The minimum length of the sequence to be generated (optional, default is 0).
- max_length: The maximum length of the sequence to be generated (optional, default is 50).
- top_p: The nucleus sampling probability (optional, default is 40).
- temperature: The temperature for sampling (optional, default is 0.6).
For example, you can use curl to send a request:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Once upon a time"}' http://localhost:5000/generate
```
The response will be a JSON object with the key generated_text, which contains the generated text.

3. Running the stream api

To start the WebSocket server, run the stream_api.py script:
```
python stream_api.py
```
This will start a WebSocket server that listens for incoming connections on port 5005.

Note: You can adjust the configuration of the server by modifying the stream_api.py script. This includes changing the port the server listens on, the path to the model files, and the parameters for the text generation.

4. Running the Client Script

```
python stream_api_client_example.py
```

Once the server is running, you can connect to it using a WebSocket client. Send your prompts as text through the WebSocket connection.
The server will return a stream of responses. Each response is a JSON object that includes the event type (either 'text_stream' or 'stream_end'), a message number, and the text generated so far (for 'text_stream' events).
Here's an example of a response:

```
{
    'event': 'text_stream',
    'message_num': 0,
    'text': 'This is the start of the generated text...'
}
```

## Acknowledgements

This project has benefited greatly from the open-source community. In particular, I'd like to acknowledge the [Text-Generation-WebUI](https://github.com/oobabooga/text-generation-webui) repository by [oobabooga](https://github.com/oobabooga). 

The implementation of the Stream API and the handling of stopping strings in the project are directly inspired by and adapted from the aforementioned repository. The clear and comprehensive structure of the repository provided valuable guidance for this project development.

I appreciate the efforts of all open-source contributors and maintainers and grateful to [oobabooga](https://github.com/oobabooga) for making their work available to the community.







