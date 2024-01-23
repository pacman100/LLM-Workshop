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

import os
import pandas as pd
import gzip
import json
from datasets import Dataset

DATAFOLDER = "hf_stack"
HF_DATASET_NAME = "hug_stack"


def load_gzip_jsonl(file_path):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_hf_dataset():
    df = None
    for file in os.listdir(DATAFOLDER):
        data = load_gzip_jsonl(os.path.join(DATAFOLDER, file))
        if df is None:
            df = pd.DataFrame(data)
        else:
            df = pd.concat([df, pd.DataFrame(data)])

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(HF_DATASET_NAME, private=False)


if __name__ == "__main__":
    create_hf_dataset()
