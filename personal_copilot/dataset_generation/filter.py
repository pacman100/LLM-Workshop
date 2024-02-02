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

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


def get_basic_stats(text):
    line_lengths = [len(line) for line in text.split("\n")]
    max_line_length = max(line_lengths)
    mean_line_length = sum(line_lengths) / len(line_lengths)
    alphanum_count = sum(char.isalpha() or char.isdigit() for char in text)
    alphanum_ratio = alphanum_count / len(text)
    return max_line_length, mean_line_length, alphanum_ratio


class BasicCodeFilter(BaseFilter):
    name = "🧑🏽‍💻 Code Filter"

    def __init__(
        self,
        max_line_length_threshold: int | None = 1000,
        mean_line_length_threshold: int | None = 100,
        alphanum_threshold: float | None = 0.25,
        exclusion_writer: DiskWriter = None,
    ):  # TODO better tune
        """ """
        super().__init__(exclusion_writer)
        self.max_line_length_threshold = max_line_length_threshold
        self.mean_line_length_threshold = mean_line_length_threshold
        self.alphanum_threshold = alphanum_threshold

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Applies heuristic rules to decide if a document should be REMOVED
        Args:
            doc

        Returns:
            False if sample.text has lines longer than max line length threshold or
            mean line length threshold or the fraction of alphanumeric charaters is less than the given threshold
        """
        text = doc.text
        filepath = doc.metadata["file_path"]
        keep_sample = True
        if text == "remove":
            keep_sample = False
        elif "ipynb" not in filepath:
            max_line_length, mean_line_length, alphanum_ratio = get_basic_stats(text)
            if (
                max_line_length > self.max_line_length_threshold
                or mean_line_length > self.mean_line_length_threshold
                or alphanum_ratio < self.alphanum_threshold
            ):
                keep_sample = False
        return keep_sample
