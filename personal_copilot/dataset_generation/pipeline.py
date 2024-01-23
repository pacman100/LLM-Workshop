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

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from reader import PersonalCopilotDatasetReader
from filter import BasicCodeFilter

MIRROR_DIRECTORY = "hf_public_repos"
TOTAL_TASKS = 16

# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    use_64bit_hashes=True
)  # better precision -> fewer false positives (collisions)


def run_code_dataset_generation():
    # stage 0 reads the code data and does basic filtering
    pipeline_0 = [
        PersonalCopilotDatasetReader(data_folder=MIRROR_DIRECTORY),
        BasicCodeFilter(),
        JsonlWriter(output_folder="filtered_data"),
    ]

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    pipeline_1 = [
        JsonlReader("filtered_data"),
        MinhashDedupSignature(
            output_folder="signatures",
            config=minhash_config,
        ),
    ]

    # stage 2 finds matches between signatures in each bucket
    pipeline_2 = [
        MinhashDedupBuckets(
            input_folder="signatures",
            output_folder="buckets",
            config=minhash_config,
        ),
    ]

    # stage 3 creates clusters of duplicates using the results from all buckets
    pipeline_3 = [
        MinhashDedupCluster(
            input_folder="buckets",
            output_folder="remove_ids",
            config=minhash_config,
        ),
    ]

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    pipeline_4 = [
        JsonlReader("filtered_data"),
        TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(
            input_folder="remove_ids",
            exclusion_writer=JsonlWriter("removed"),
        ),
        JsonlWriter(output_folder="hf_stack"),
    ]

    executor_0: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_0, tasks=TOTAL_TASKS
    )

    executor_1: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_1, tasks=TOTAL_TASKS
    )

    executor_2: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_2,
        tasks=minhash_config.num_buckets,
    )

    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, tasks=1)

    executor_4: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_4, tasks=TOTAL_TASKS
    )

    print(executor_0.run())
    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())
    print(executor_4.run())


if __name__ == "__main__":
    run_code_dataset_generation()
