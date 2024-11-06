# Getting Started with I2-HOFI
This document provides a brief intro of launching jobs in I2-HOFI for training and testing. Before launching any job, make sure you have properly installed the required packages following the instruction in README.md

- Download the extract the github source files to ./path_to_source_dir/I2-HOFI-main

- Update the `configs/config_DATASET_NAME.yaml` config file with your API key of WandB if you want to register `train` and `validation` performance metrics to WandB 

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

## GPU Settings
- For running on CPU (default config)
```python
python hofi/train.py dataset DATASET_NAME gpu_id -1
```
- For running in GPU (if you have single GPU, your gpu_id is 0, if you have multiple GPU (say 3) then select gpu_id as 0/1/2 and so on)
```python
python hofi/train.py dataset DATASET_NAME gpu_id 0 gpu_utilisation 0.8
```

## Other Configuration setting
- To set batch size of 16 from console or modify directly on config files
 ```python
python hofi/train.py dataset DATASET_NAME batch_size 16
```


