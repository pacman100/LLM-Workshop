# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import os
import warnings
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import list_repo_files
from huggingface_hub.utils._validators import HFValidationError
from peft import PeftConfig, PeftModel

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        prompt_batch, chosen_batch, rejected_batch = [], [], []
        for chosen, rejected in zip(samples["chosen"], samples["rejected"]):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = chosen[:-1]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = chosen[-1:]
            rejected_messages = rejected[-1:]
            chosen_batch.append(
                tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            )
            rejected_batch.append(
                tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            )
            prompt_batch.append(
                tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            )
        return {
            "text_chosen": chosen_batch,
            "text_rejected": rejected_batch,
            "text_prompt": prompt_batch,
        }

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(
                f"Split type {split} not recognized as one of test or train."
            )

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = (
            int(os.environ.get("LOCAL_RANK", -1))
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "auto"
        )  # {"": 0}

    # assumes the tokenizer has special tokens and chat template post SFT
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    model_kwargs = {
        "load_in_8bit": load_in_8bit,
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if args.use_flash_attn else "eager",
    }

    peft_config = None
    if is_adapter_model(args.model_name_or_path) is True:
        if not args.use_peft_lora:
            warnings.warn(
                "Setting `use_peft_lora` to `True` as the SFT model is a PEFT model."
            )
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, **model_kwargs
        )
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model = PeftModel.from_pretrained(
            model, args.model_name_or_path, is_trainable=True, adapter_name="default"
        )
        # Load the adapter a second time, with a different name, which will be our reference model.
        model.load_adapter(args.model_name_or_path, adapter_name="reference")
    else:
        model = args.model_name_or_path
        if args.use_peft_lora:
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.lora_target_modules.split(","),
            )

    return model, peft_config, tokenizer, model_kwargs


# copied below function from
# https://github.com/huggingface/alignment-handbook/blob/cbcb3f60fbc8b8884e15e181ff49e9549ec5df00/src/alignment/model_utils.py#L101
def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except HFValidationError:
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return (
        "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files
    )
