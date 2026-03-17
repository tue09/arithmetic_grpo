# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the historical AIME dataset to parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "gneubig/aime-1983-2024"
INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."
KEY_CANDIDATES = {
    "year": ["Year", "year"],
    "problem_number": ["Problem Number", "Problem_Number", "problem_number"],
    "question": ["Problem", "Question", "problem", "question"],
    "answer": ["Answer", "answer", "solution"],
}


def find_key(column_names, key_name):
    for candidate in KEY_CANDIDATES[key_name]:
        if candidate in column_names:
            return candidate
    raise KeyError(f"Could not find {key_name} column in dataset columns: {column_names}")


def load_raw_dataset(local_dataset_path=None):
    if local_dataset_path is None:
        dataset = datasets.load_dataset(DATA_SOURCE)
    elif os.path.isdir(local_dataset_path):
        dataset = datasets.load_dataset(local_dataset_path)
    elif local_dataset_path.endswith(".csv"):
        dataset = datasets.load_dataset("csv", data_files=local_dataset_path)
    elif local_dataset_path.endswith(".json") or local_dataset_path.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=local_dataset_path)
    else:
        dataset = datasets.load_dataset(local_dataset_path)

    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return next(iter(dataset.values()))

    return dataset


def build_dataset(raw_dataset, split_name, year_min=None, year_max=None, data_source="aime_history"):
    year_key = find_key(raw_dataset.column_names, "year")
    problem_number_key = find_key(raw_dataset.column_names, "problem_number")
    question_key = find_key(raw_dataset.column_names, "question")
    answer_key = find_key(raw_dataset.column_names, "answer")

    def year_in_range(example):
        year = int(example[year_key])
        if year_min is not None and year < year_min:
            return False
        if year_max is not None and year > year_max:
            return False
        return True

    filtered_dataset = raw_dataset.filter(year_in_range)

    def process_fn(example, idx):
        year = int(example[year_key])
        problem_number = int(example[problem_number_key])
        question = str(example[question_key]).strip()
        ground_truth = str(example[answer_key]).strip()
        prompt = f"{question} {INSTRUCTION_FOLLOWING}"
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "year": year,
                "problem_number": problem_number,
                "question": question,
            },
        }

    return filtered_dataset.map(process_fn, with_indices=True, remove_columns=filtered_dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/aime",
        help="The save directory for the preprocessed dataset.",
    )
    args = parser.parse_args()

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading the {DATA_SOURCE} dataset from huggingface...", flush=True)
    raw_dataset = load_raw_dataset(args.local_dataset_path)

    dataset_all = build_dataset(raw_dataset, split_name="train", year_min=1983, year_max=2024)
    dataset_until_2022 = build_dataset(raw_dataset, split_name="train", year_min=1983, year_max=2022)
    dataset_until_2023 = build_dataset(raw_dataset, split_name="train", year_min=1983, year_max=2023)
    dataset_2023 = build_dataset(raw_dataset, split_name="test", year_min=2023, year_max=2023, data_source="aime23")
    dataset_2024 = build_dataset(raw_dataset, split_name="test", year_min=2024, year_max=2024, data_source="aime24")

    outputs = [
        ("aime-history-1983-2024.parquet", "aime-history-1983-2024.example.json", dataset_all),
        ("aime-history-1983-2022.parquet", "aime-history-1983-2022.example.json", dataset_until_2022),
        ("aime-history-1983-2023.parquet", "aime-history-1983-2023.example.json", dataset_until_2023),
        ("aime-history-2023.parquet", "aime-history-2023.example.json", dataset_2023),
        ("aime-history-2024.parquet", "aime-history-2024.example.json", dataset_2024),
    ]

    for parquet_name, example_name, dataset in outputs:
        dataset.to_parquet(os.path.join(local_dir, parquet_name))
        if len(dataset) > 0:
            with open(os.path.join(local_dir, example_name), "w") as f:
                json.dump(dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
