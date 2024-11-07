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

-----------------------

# Getting Started with I2-HOFI

This document provides an introduction on how to launch training and testing jobs using I2-HOFI. Ensure that all required packages are installed by following the instructions provided in the README.md file.

### Setup Instructions

1. **Clone and Set Up the Repository:**
   - Clone the repository and extract the source files to `./path_to_source_dir/I2-HOFI-main`.

2. **Configure Settings:**
   - Update the `configs/config_DATASET_NAME.yaml` configuration file with your WandB API key to enable logging of `training` and `validation` metrics in WandB.

3. **Prepare Data:**
   - Follow the instructions in [DATASET.md](datasets/DATASET.md) to download and extract the datasets into the specified directories.

4. **Navigate to the Project Directory:**
   - Open your terminal and change the directory to the root of the project:
     ```bash
     cd ./path_to_root_dir/I2-HOFI-main/
     ```

### Training the I2HOFI Model

To start training the model, execute the following command in the terminal:

```bash
python hofi/train.py dataset DATASET_NAME
```
Replace `DATASET_NAME` with one of the following dataset identifiers, noting that they are case-sensitive: `Aircraft`, `Cars`, `CUB200`, `Flower102`, `NABird`.

### GPU Configuration

- **For CPU Usage:**
  - Run the following command to train using CPU:
    ```bash
    python hofi/train.py dataset DATASET_NAME gpu_id -1
    ```

- **For GPU Usage:**
  - For a single GPU setup (if you have one GPU, set `gpu_id` to 0) with 80% memory allocation, use the following command:
    ```python
    python hofi/train.py dataset DATASET_NAME gpu_id 0 gpu_utilisation 0.8
    ```
  - For multiple GPUs, specify the GPU ID (0, 1, 2, etc.) and set memory allocation to 80% using this command:
    ```python
    python hofi/train.py dataset DATASET_NAME gpu_id 0 gpu_utilisation 0.8
    ```
    Replace `0` with the appropriate GPU ID as needed.

### Additional Configuration Settings

- **Set Batch Size:**
  - To configure the batch size to 16 via the command line (or modify directly in the config files):
    ```bash
    python hofi/train.py dataset DATASET_NAME batch_size 16
    ```

Ensure that each step is followed correctly to facilitate a smooth setup and execution of training jobs in I2-HOFI.
