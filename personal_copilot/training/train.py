"""
Fine-Tune StarCoder on code/text dataset
"""

import argparse
import os
import random
import subprocess
import warnings

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

import fim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/starcoderplus")
    parser.add_argument("--dataset_name", type=str, default="smangrul/hf-stack-v1")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--test_size", type=float, default=0.005)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument("--fim_rate", type=float, default=0)
    parser.add_argument("--fim_spm_rate", type=float, default=0)

    parser.add_argument("--use_peft_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--use_flash_attn", action="store_true")

    parser.add_argument("--use_4bit_qunatization", action="store_true")
    parser.add_argument("--use_nested_quant", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")

    parser.add_argument("--use_8bit_qunatization", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = fim.get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )

    return train_dataset, valid_dataset


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
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.no_gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attn,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
    )

    if (args.use_4bit_qunatization or args.use_8bit_qunatization) and args.use_peft_lora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.no_gradient_checkpointing)

    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )

        if args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def run_training(args, train_data, val_data):
    train_data.start_iteration = 0

    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true" and args.use_peft_lora
    )

    save_strategy = "no" if is_deepspeed_peft_enabled else "steps"

    print(f"Starting main loop")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
        fp16=args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=f"starcoder-copilot",
        push_to_hub=args.push_to_hub,
        include_tokens_per_second=True,
    )

    print("Loading the model")
    model = create_and_prepare_model(args)
    print(model)
    if args.use_peft_lora:
        model.print_trainable_parameters()

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data)

    # post process for faster training when using PEFT + INT4 Quantization
    if args.use_peft_lora:
        for name, module in trainer.model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
                if hasattr(module, "weight"):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    print("Training...")
    trainer.train()
    if args.use_peft_lora:
        print("Saving last checkpoint of the model")
        model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))

    if is_deepspeed_peft_enabled:
        trainer.accelerator.wait_for_everyone()
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        unwrapped_model.save_pretrained(
            args.output_dir, state_dict=trainer.accelerator.get_state_dict(trainer.deepspeed)
        )
        trainer.accelerator.wait_for_everyone()

    # Save model and tokenizer
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.save_model(args.output_dir)
    trainer.accelerator.print(f"Model saved to {args.output_dir}")
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

    if args.push_to_hub:
        trainer.model.push_to_hub(args.output_dir)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True, trust_remote_code=True)

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
