from dataclasses import dataclass, field

from transformers import HfArgumentParser

# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import load_state_dict


def shard_checkpoint(checkpoint_dir: str, shard_size_gb: int = 10):
    checkpoint_file = f"{checkpoint_dir}/pytorch_model.bin"
    if os.path.exists(checkpoint_file):
        file_size_gb = os.path.getsize(checkpoint_file) / (1024 * 1024 * 1024)
        if file_size_gb > shard_size_gb:
            print(
                f"`pytorch_model.bin` is greater than {shard_size_gb}GB, sharding the model in {shard_size_gb}GB chunks"
            )
            print("loading the checkpoint")
            # We load on CPU to avoid OOM errors on GPU
            state_dict = load_state_dict(checkpoint_file)
            print("checkpoint loaded")
            config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
            print("loading checkpoint into the model")
            model = AutoModelForCausalLM.from_pretrained(
                None, config=config, state_dict=state_dict, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            print("model loaded with the checkpoint")
            print("sharding the model")
            model.save_pretrained(checkpoint_dir, max_shard_size=f"{shard_size_gb}GB")
            print("model sharded")
            os.remove(checkpoint_file)
            print("`pytorch_model.bin` deleted.")


@dataclass
class ScriptArguments:
    output_dir: str = field(metadata={"help": "The directory where the chckpoint is stored"})
    shard_size_gb: int = field(default=10, metadata={"help": "The size of each shard in GB"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    shard_checkpoint(args.output_dir, shard_size_gb=args.shard_size_gb)


if __name__ == "__main__":
    main()
