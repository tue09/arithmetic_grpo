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
Preprocess the AIME 2024 / 2025 datasets to parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."

DATASET_SPECS = {
    "aime24": {
        "hf_name": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "question_key": "Problem",
        "answer_key": "Answer",
        "output_name": "aime-2024.parquet",
        "example_name": "aime-2024.example.json",
        "data_source": "aime24",
    },
    "aime25": {
        "hf_name": "yentinglin/aime_2025",
        "split": "train",
        "question_key": "problem",
        "answer_key": "solution",
        "output_name": "aime-2025.parquet",
        "example_name": "aime-2025.example.json",
        "data_source": "aime25",
    },
}


def build_dataset(dataset_name, local_dataset_path=None):
    spec = DATASET_SPECS[dataset_name]

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, split=spec["split"])
    else:
        dataset = datasets.load_dataset(spec["hf_name"], split=spec["split"])

    def process_fn(example, idx):
        question = str(example[spec["question_key"]]).strip()
        ground_truth = str(example[spec["answer_key"]]).strip()
        prompt = f"{question} {INSTRUCTION_FOLLOWING}"
        return {
            "data_source": spec["data_source"],
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": "test",
                "index": idx,
                "question": question,
            },
        }

    return dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=["aime24", "aime25"])
    parser.add_argument("--aime24_local_dataset_path", default=None)
    parser.add_argument("--aime25_local_dataset_path", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/aime",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    selected_datasets = []
    for dataset_name in args.datasets:
        if dataset_name == "all":
            selected_datasets.extend(["aime24", "aime25"])
            continue
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        selected_datasets.append(dataset_name)

    selected_datasets = list(dict.fromkeys(selected_datasets))

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    dataset_path_overrides = {
        "aime24": args.aime24_local_dataset_path,
        "aime25": args.aime25_local_dataset_path,
    }

    for dataset_name in selected_datasets:
        spec = DATASET_SPECS[dataset_name]
        print(f"Loading the {spec['hf_name']} dataset from huggingface...", flush=True)
        dataset = build_dataset(dataset_name, dataset_path_overrides[dataset_name])
        dataset.to_parquet(os.path.join(local_dir, spec["output_name"]))
        with open(os.path.join(local_dir, spec["example_name"]), "w") as f:
            json.dump(dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
