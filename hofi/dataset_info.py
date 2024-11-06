def datasetInfo(dataset_name):  
    if dataset_name == "Cars":
        nb_classes  = 196
    elif dataset_name == "Aircraft":
        nb_classes  = 100
    elif dataset_name == "CUB200":
        nb_classes  = 200
    elif dataset_name == "Dogs":
        nb_classes  = 120
    elif dataset_name == "Flower102":
        nb_classes  = 102
    elif dataset_name == "NABird":
        nb_classes  = 555
    elif dataset_name == "PPMI":
        nb_classes  = 24
    elif dataset_name == "Stanford40_bb":
        nb_classes  = 40
    elif dataset_name == "Food_101":
        nb_classes = 101
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are : Cars, Aircraft")
    
    # Add more elif statements for other datasets as needed

    return nb_classes
