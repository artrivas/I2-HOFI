# Getting Started with I2-HOFI
This document provides a brief intro of launching jobs in I2-HOFI for training and testing. Before launching any job, make sure you have properly installed the required packages following the instruction in README.md

- Download the extract the github source files to ./path_to_source_dir/I2-HOFI-main

- Download and extract datasets into the required folder based on the instruction given in [DATASET.md](datasets/DATASET.md)

- First change the path to  the root directory from the console
  ``` bash
  ./path_to_root_dir/I2-HOFI-main/
  ```

## Train the I2HOFI Model
Here we can start training model by typing the following python command in the console 

```python
python hofi/train.py dataset DATASET_NAME
```

DATASET_NAME can be either of `Aircraft`, `Cars`, `CUB200`, `Flower102`, `NABird`

Please note the Dataset name is CASE SENSITIVE
