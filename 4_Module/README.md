# Finetuning and Prompt Engineering

## Prompting

1. Go to terminal and run:
```bash
vim .env
```
2. Copy the HF TOKEN and save:
```
TOKEN={HF TOKEN}
```
3. install requirements:
```
pip install -r requirements.txt
```
4. Go to [Prompt Engineering Notebook](../4_Module/Prompt_Engineering.ipynb)
5. Go to [QA Bot Code using ChatGPT](https://chat.openai.com/share/eb3079ba-1379-4b9a-b21c-839feb023309) for seeing how I used ChatGPT to scrape the PEFT documentation using `scrapy` and `BeautifulSoap`, chunk it, embed the chunks using `sentence-transformers`, create index using `hnswlib` and loading the search index and utils for embedding user query.

## Finetuning StarCoder on your private codebase to get personal co-pilot

### Dataset Generation
Go to [Dataset Generation](../personal_copilot/dataset_generation/) folder to seee how to create dataset from internal codebase for training your own co-pilot on internal codebase.

### Training

1. Go to [personal_copilot](../personal_copilot/) and install requirements
```
pip install -r requirements.txt
```
2. Go to [train.py](../personal_copilot/training/train.py) for the training code using 🤗 Accelerate and 🤗 Transformers Trainer.  
3. Go to [run_deepspeed.sh](../personal_copilot/training/run_deepspeed.sh) to fully finetune `starcoderbase-3b` model with ZeRO Stage-3 and CPU offloading.
4. Infere using the trained model in this notebook.
5. Go to [run_fsdp.sh](../personal_copilot/training/run_fsdp.sh) to fully finetune `starcoderbase-3b` model with FSDP when atleast 4 GPUs are available.