## Instruction to prepare datasets

## Dataset Prepration
Download the dataset and place into the datasets folder. You can download from the offcial website or directly download the organized dataset zip file from below repository.

### Aircraft Dataset:
- Official link: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- Our Repo link: [Aircraft Dataset](https://drive.google.com/uc?export=download&id=1v_cOB1gOIneI-Y1vJC7WUSvwH2FP9qCS)

### Caltech-UCSD Birds (CUB-200):
- Official link: https://data.caltech.edu/records/65de6-vp158
- Our Repo link:

### Stanford Cars dataset
- Official link:
- Our Repo link:

### NABirds dataset:
- Official link:
- Our Repo link:

### Oxford Flowers dataset:
- Official link:
- Our Repo link:
  

For example, after extracting the Aircraft dataset, your train and test subdirectory will look below
```bash
./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/train/...

./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/test/...
```
After you download multiple datasets, say DATASET1 and DATASET2. Your Directory structure will look something like below
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
