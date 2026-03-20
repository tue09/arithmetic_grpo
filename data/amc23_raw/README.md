---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 25709
    num_examples: 40
  download_size: 18688
  dataset_size: 25709
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
