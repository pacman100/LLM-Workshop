import random
import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl.trainer import PackedIterableDataset, ConstantLengthDataset


def save_peft_deepspeed_ckpt(trainer, output_dir):
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            save_peft_deepspeed_ckpt(self.trainer, args.output_dir)
        return control


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def create_datasets(tokenizer, data_args, training_args):
    dataset = load_dataset(
        data_args.dataset_name,
        use_auth_token=True,
        num_proc=training_args.dataloader_num_workers,
    )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(
        train_data, tokenizer, data_args.dataset_text_field
    )
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    dataset_class = (
        ConstantLengthDataset
        if "epoch" in str(training_args.save_strategy)
        else PackedIterableDataset
    )

    train_dataset = dataset_class(
        tokenizer,
        train_data,
        seq_length=data_args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=data_args.dataset_text_field,
        shuffle=True,
        append_concat_token=False,
    )
    valid_dataset = dataset_class(
        tokenizer,
        valid_data,
        seq_length=data_args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=data_args.dataset_text_field,
        shuffle=False,
        append_concat_token=False,
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
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_flash_attention_2=args.use_flash_attn,
        trust_remote_code=True,
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
