## Instruction to prepare datasets

## Dataset Prepration
Download the dataset and place into the datasets folder. You can download from the offcial website or from the zip file from below repository. 
- [Aircraft Dataset](https://drive.google.com/uc?export=download&id=1v_cOB1gOIneI-Y1vJC7WUSvwH2FP9qCS)
  
For example, after extracting the Aircraft dataset, your train and test subdirectory will look below
```bash
./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/train/...

./path_to_source_dir/I2-HOFI-main/datasets/Aircraft/test/...
```

```bash
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```
