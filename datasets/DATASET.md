## INSTRUCTIONs TO PREPARE DATASETS

## Dataset Prepration
Download the dataset and place into the datasets folder. You can download from the offcial website or directly download the organized dataset zip file from below repository in organized format to directly train and test on our code.

### Aircraft Dataset:
- Official link: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- Our Repo link: [Aircraft Dataset](https://drive.google.com/uc?export=download&id=1v_cOB1gOIneI-Y1vJC7WUSvwH2FP9qCS)

### Caltech-UCSD Birds (CUB-200):
- Official link: https://data.caltech.edu/records/65de6-vp158
- Our Repo link: [CUB200 Dataset](https://drive.google.com/uc?export=download&id=1S9RgrN-Ys6Ogc11av-9apy9sMeMuoqDZ)

### Stanford Cars dataset
- Official link: https://pytorch.org/vision/0.16/generated/torchvision.datasets.StanfordCars.html
- Our Repo link: [Cars Dataset](https://drive.google.com/uc?export=download&id=1DhVbnAlBaY75n6YNbyopwyPulkjszk-m)

### NABirds dataset:
- Official link: https://dl.allaboutbirds.org/nabirds
- Our Repo link: [NABirds Dataset](https://drive.google.com/uc?export=download&id=1B7eYvXTXNGrJcMDySU62U-RGXF9b-5zh)

### Oxford Flowers dataset:
- Official link: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Our Repo link: [Flowers Dataset](https://drive.google.com/uc?export=download&id=10fFJGlCAE1NC5eGoun4nW6C6s_CpBKEH)
 

For example, after extracting the Aircraft dataset, your train and test subdirectory will look below
```bash
./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/train/...

./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/test/...
```

If you are downloading from the official website, you need to organize the dataset into the following specific directory structure to run on our code. Also after you download multiple datasets, say DATASET1 and DATASET2, your Directory structure will look something like below
```bash
datasets
|_ DATASET1
|  |_ train
|  |  |_ train_folder1
|  |  |_ train_folder2
|  |  |_ ...
|  |_ test
|     |_ test_folder1
|     |_ test_folder2
|     |_ ...
|
|_ DATASET2
|  |_ train
|  |  |_ train_folder1
|  |  |_ train_folder2
|  |  |_ ...
|  |_ test
|     |_ test_folder1
|     |_ test_folder2
|     |_ ...
|
|_ ...
```
Please let us know if you are facing any issue with dataset download from our repo link





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

After downloading and extracting a dataset, organize it into the following structure to ensure compatibility with our code:

```bash
datasets/
├── DATASET1/
│   ├── train/
│   │   ├── train_folder1/
│   │   ├── train_folder2/
│   │   └── ...
│   └── test/
│       ├── test_folder1/
│       ├── test_folder2/
│       └── ...
├── DATASET2/
│   ├── train/
│   │   ├── train_folder1/
│   │   ├── train_folder2/
│   │   └── ...
│   └── test/
│       ├── test_folder1/
│       ├── test_folder2/
│       └── ...
└── ...

