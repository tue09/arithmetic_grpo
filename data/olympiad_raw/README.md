---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- olympiad
- mathematics
- competition
size_categories:
- 1K<n<10K
---

# OlympiadBench (Split Version)

This dataset is a split version of the original [knoveleng/OlympiadBench](https://huggingface.co/datasets/knoveleng/OlympiadBench) dataset.

## Dataset Description

This dataset contains mathematical olympiad problems with their solutions, split into training and test sets.

## Dataset Structure

- **Train split**: 575 examples
- **Test split**: 100 examples (last 100 examples from original dataset)

### Data Fields

- `question`: The olympiad problem statement
- `answer`: The solution to the problem

## Usage

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("weijiezz/OlympiadBench-split")

# Load specific splits
train_dataset = load_dataset("weijiezz/OlympiadBench-split", split="train")
test_dataset = load_dataset("weijiezz/OlympiadBench-split", split="test")
```

## Source

This dataset is derived from [knoveleng/OlympiadBench](https://huggingface.co/datasets/knoveleng/OlympiadBench).

## License

MIT License (following the original dataset)
