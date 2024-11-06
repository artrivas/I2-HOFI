# Getting Started with I2-HOFI
This document provides a brief intro of launching jobs in I2-HOFI for training and testing. Before launching any job, make sure you have properly installed the required packages following the instruction in README.md

- Download the extract the github source files to ./path_to_source_dir/I2-HOFI-main

- First navigate to the source directory from the console
  ``` bash
  cd ./path_to_source_dir/I2-HOFI-main
  ```

## Train a Standard Model from Scratch
Here we can start with training a 

```python
python hofi/train.py dataset DATASET_NAME
```

DATASET_NAME can be either of `Aircraft`, `Cars`

Please note the Dataset name is CASE SENSITIVE
