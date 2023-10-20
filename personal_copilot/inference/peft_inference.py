import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import Optional
import contextlib

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

model = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# model = model.merge_and_unload()
if not hasattr(model, "hf_device_map"):
    model.cuda()

model_id = "smangrul/peft-lora-starcoder15B-v2-personal-copilot-A100-40GB-colab"
model = PeftModel.from_pretrained(model, model_id, adapter_name="personal_copilot")
model.add_weighted_adapter(["personal_copilot"], [0.8], "best_personal_copilot")
model.set_adapter("best_personal_copilot")


def get_code_completion(prefix, suffix):
    text = prompt = f"""<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"""
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
