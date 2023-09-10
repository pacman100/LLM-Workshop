# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
import os
import subprocess
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, Trainer
from utils import *

########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="Salesforce/codegen25-7b-multi",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=10, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(default="results", metadata={"help": "Where to store the final model."})
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(default=4, metadata={"help": "Number of dataset workers to use."})
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tests things like proper saving/loading/logging of model"},
    )


def main(args):
    # training arguments
    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true" and args.use_peft_lora
    )
    save_strategy = "no" if is_deepspeed_peft_enabled else "steps"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # model
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    # trainer
    trainer = Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.accelerator.print(f"{trainer.model}")
    if args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if is_deepspeed_peft_enabled:
        trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=args.save_steps))

    if args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, args)

    # train
    trainer.train()

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if is_deepspeed_peft_enabled:
        trainer.accelerator.wait_for_everyone()
        state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        if trainer.accelerator.is_main_process:
            unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        trainer.accelerator.wait_for_everyone()
    else:
        if args.push_to_hub:
            trainer.push_to_hub()
            if args.use_peft_lora:
                trainer.model.push_to_hub(args.output_dir)
        else:
            trainer.save_model(args.output_dir)

    # Save everything else on main process
    if trainer.args.process_index == 0:
        print("Sharding model if >10GB...")
        # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
        # We run this in a subprocess to avoid interference from the accelerators.
        subprocess.run(
            [
                "python",
                "shard_checkpoint.py",
                f"--output_dir={args.output_dir}",
            ],
            check=True,
        )
        os.remove(os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
