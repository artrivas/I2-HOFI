# Dataset Preparation Instructions

Follow these steps to download and organize the datasets required for training and testing the models. You have the option to download directly from the official sources or use organized versions available through provided repository links.

## Available Datasets
Below are the datasets you can download and their respective sources:

### 1. Aircraft Dataset
- **Official link:** [Aircraft Dataset Official](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- **Repository link:** [Download Aircraft Dataset](https://drive.google.com/uc?export=download&id=1v_cOB1gOIneI-Y1vJC7WUSvwH2FP9qCS)

### 2. Caltech-UCSD Birds (CUB-200)
- **Official link:** [CUB-200 Official](https://data.caltech.edu/records/65de6-vp158)
- **Repository link:** [Download CUB200 Dataset](https://drive.google.com/uc?export=download&id=1S9RgrN-Ys6Ogc11av-9apy9sMeMuoqDZ)

### 3. Stanford Cars Dataset
- **Official link:** [Stanford Cars Official](https://pytorch.org/vision/0.16/generated/torchvision.datasets.StanfordCars.html)
- **Repository link:** [Download Cars Dataset](https://drive.google.com/uc?export=download&id=1DhVbnAlBaY75n6YNbyopwyPulkjszk-m)

### 4. NABirds Dataset
- **Official link:** [NABirds Official](https://dl.allaboutbirds.org/nabirds)
- **Repository link:** [Download NABirds Dataset](https://drive.google.com/uc?export=download&id=1B7eYvXTXNGrJcMDySU62U-RGXF9b-5zh)

### 5. Oxford Flowers Dataset
- **Official link:** [Oxford Flowers Official](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Repository link:** [Download Flowers Dataset](https://drive.google.com/uc?export=download&id=10fFJGlCAE1NC5eGoun4nW6C6s_CpBKEH)

## Dataset Structure
After downloading and extracting a dataset, please organize it as shown below to ensure compatibility with our code. For example, if you have two datasets, such as `Aircraft` and `Stanford Cars`, they should be arranged into their respective `train` and `test` subdirectories within the `datasets` folder:

```bash
datasets/
|_ Aircraft/
|  |_ train/
|  |  |_ train_folder1/
|  |  |_ train_folder2/
|  |  |_ ...
|  |_ test/
|     |_ test_folder1/
|     |_ test_folder2/
|     |_ ...
|
|_ Cars/
|  |_ train/
|  |  |_ train_folder1/
|  |  |_ train_folder2/
|  |  |_ ...
|  |_ test/
|     |_ test_folder1/
|     |_ test_folder2/
|     |_ ...
|
|_ ...
```
This structure should be applied to each dataset you use, replacing Aircraft and Stanford Cars with the actual names of the datasets you are working with. 

If you encounter difficulties while downloading datasets from our repository, or if you have suggestions for improving the download process, please feel free to create an issue in this repository. Your detailed feedback is essential for us to make necessary adjustments and enhance the usability of our code.
