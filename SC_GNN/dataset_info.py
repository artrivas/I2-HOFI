import sys
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def datasetInfo(dataset_name):
    
    if dataset_name == "Cars":
        nb_classes  = 196
    elif dataset_name == "Aircraft":
        nb_classes  = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are : Cars, Aircraft")
    
    # Add more elif statements for other datasets as needed

    return nb_classes
