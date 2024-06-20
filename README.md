# Mixtral 8x7B / 8x22B chat

This repository is intended to run the Mistral Mixtrals models as worker for the [AIME API Server](https://github.com/aime-team/aime-api-server). An interactive console chat for testing purpose is available as well.

Mixtral demo server running at: [https://api.aime.info/mixtral-chat/](https://api.aime.info/mixtral-chat/)

## Installation

Note: You will need a GPU to install it as it currently requires `xformers` to be installed and `xformers` itself needs a GPU for installation.

### Download Model

| Name        | Download | md5sum |
|-------------|-------|-------|
| 7B Instruct | https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar | `80b71fcb6416085bcb4efad86dfb4d52` |
| 8x7B Instruct | https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar (**Updated model coming soon!**) | `8e2d3930145dc43d3084396f49d38a3f` |
| 8x22 Instruct | https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-Instruct-v0.3.tar | `471a02a6902706a2f1e44a693813855b` |
| 7B Base | https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar | `0663b293810d7571dad25dae2f2a5806` |
| 8x7B |     **Updated model coming soon!**       | - |
| 8x22B | https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-v0.3.tar | `a2fa75117174f87d1197e3a4eb50371a` |
| Codestral 22B | https://models.mistralcdn.com/codestral-22b-v0-1/codestral-22B-v0.1.tar | `a5661f2f6c6ee4d6820a2f68db934c5d` |

Note: 
- **Important**:
  - `mixtral-8x22B-Instruct-v0.3.tar` is exactly the same as [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), only stored in `.safetensors` format
  - `mixtral-8x22B-v0.3.tar` is the same as [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), but has an extended vocabulary of 32768 tokens.
  - `codestral-22B-v0.1.tar` has a custom non-commercial license, called [Mistral AI Non-Production (MNPL) License](https://mistral.ai/licenses/MNPL-0.1.md)
- All of the listed models above supports function calling. For example, Mistral 7B Base/Instruct v3 is a minor update to Mistral 7B Base/Instruct v2,  with the addition of function calling capabilities. 
- The "coming soon" models will include function calling as well. 
- You can download the previous versions of our models from our [docs](https://docs.mistral.ai/getting-started/open_weight_models/#downloading).


Download any of the above links and extract the content, *e.g.*:

```sh
cd /destination/to/store/the/model/weights/
wget https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar
tar -xf Mixtral-8x7B-v0.1-Instruct.tar
```

### Setup AIME MLC

Easy installation within an [AIME ML-Container](https://github.com/aime-team/aime-ml-containers).

1. Create and open an AIME ML container:

```mlc-create mycontainer Pytorch 2.3.0```
Once done open the container with:
```mlc-open mycontainer```

2. Install the required pip packages in the AIME ML container.

```
pip install -r /destination/of/mixtral/repo/mixtral_chat/requirements.txt
```

### Start a Chat with Mixtral in Command Line

Run the chat mode in the command line with following command:
```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir <destination_of_checkpoints> --tokenizer_path <destination_of_tokenizer>
```
It will start a single user chat (batch_size is 1) with Chloe.

### Start Mixtral Chat as AIME API Worker

To run Mixtral Chat as a worker for the [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir /destination/to/store/the/model/weights/ --tokenizer_path /destination/to/store/the/model/weights/tokenizer.model --api_server <url to api server>
```
It will start Mixtral as worker, waiting for job request through the AIME API Server. Use the --max_batch_size option to control how many parallel job requests can be handled (depending on the available GPU memory). 


## References

[1]: [LoRA](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation of Large Language Models, Hu et al. 2021
