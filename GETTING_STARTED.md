# Getting Started with I2-HOFI

This document provides an introduction on how to launch training and testing jobs using I2-HOFI. Ensure that all required packages are installed by following the instructions provided in the [README.md](README.md) file.

### Setup Instructions

1. **Clone and Set Up the Repository:**
   - Clone the repository and extract the source files to `path_to_root_dir/I2-HOFI-main`. After extraction you will have the following directory structure:
   ```bash
   path_to_root_dir/
     |_ I2-HOFI-main/
         |_ configs/
         |_ datasets/
         |_ hofi/
         |_ media/
         |_ GETTING_STARTED.md
         |_ README.md
   ```
   Replace `path_to_root_dir` in the path `path_to_root_dir/I2-HOFI-main` with the full path where you have extracted the repository files.
2. **Configure Settings:**
   - Update the `configs/config_DATASET_NAME.yaml` configuration file with your WandB API key to enable logging of `training` and `validation` metrics in WandB.

3. **Prepare Data:**
   - Follow the instructions in [DATASET.md](datasets/DATASET.md) to download and extract the datasets into the specified directories.

4. **Navigate to the Project Directory:**
   - Open your terminal and change the directory to the root of the project:
     ```bash
     cd path_to_root_dir/I2-HOFI-main/
     ```

### Training the I2HOFI Model

To start training the model, execute the following command in the terminal:

```python
python hofi/train.py dataset DATASET_NAME
```
Replace `DATASET_NAME` with one of the following dataset identifiers, noting that they are case-sensitive: `Aircraft`, `Cars`, `CUB200`, `Flower102`, `NABird`.

### GPU Configuration

- **For CPU Usage:**
  - Run the following command to train using CPU (default setting):
    ```python
    python hofi/train.py dataset DATASET_NAME gpu_id -1
    ```

- **For GPU Usage:**
  - For a single GPU setup (if you have one GPU, set `gpu_id` to 0) with 80% memory allocation, use the following command:
    ```python
    python hofi/train.py dataset DATASET_NAME gpu_id 0 gpu_utilisation 0.8
    ```
  - For multiple GPUs, specify the GPU ID (0, 1, 2, etc.) and set memory allocation to 50% using this command:
    ```python
    python hofi/train.py dataset DATASET_NAME gpu_id 0 gpu_utilisation 0.5
    ```
    Replace `0` with the appropriate GPU ID as needed.

### Additional Configuration Settings

- **Set Batch Size:**
  - To configure the batch size to 16 via the command line (or modify directly in the config files):
    ```python
    python hofi/train.py dataset DATASET_NAME batch_size 16
    ```
    
- **Training with Different Backbone Models:**
  - You can train the I2HOFI framework with various backbone models from the `tf.keras.applications` library by adjusting the training command as follows:
    ```python
    python hofi/train.py --dataset DATASET_NAME --backbone MODEL_NAME
    ```
    Here, `MODEL_NAME` should match the exact case-sensitive name as it appears in the TensorFlow documentation. For example:
   - For resnet50, use:
     ```python
     python hofi/train.py --dataset DATASET_NAME --backbone ResNet50
     ```
     See [tf.keras.applications.ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50).

   - For densenet121, use:
     ```python
     python hofi/train.py --dataset DATASET_NAME --backbone DenseNet121
     ```
     See [tf.keras.applications.DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121).

Refer to the [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) for a complete list of available models. The I2-HOFI model requires backbone models with channel dimensions in multiples of `2^n`, where `n` is an integer. For models with different channel dimensions, add a convolutional layer to adjust them to the nearest multiple of `2^n`. Note that optimal configuration settings may vary between different backbone models compared to Xception.

Ensure that each step is followed correctly to facilitate a smooth setup and execution of training jobs in I2-HOFI.
