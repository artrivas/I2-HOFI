
from tensorflow.keras.models import Model
# from .models0 import *
from .models import *


""" Importing model definations """
model_classes = {
    "scgnn": SC_GNN,
    # Add more models
    "scgnn_intra": SC_GNN_INTRA,
    # "scgnn_inter": SC_GNN_INTER,
    "scgnn_comb": SC_GNN_COMB,

    # Models with GATconv
    "scgnnres_comb": SC_GNNRES_COMB,
    "scgnngatres_comb": SC_APPNPGATRES_COMB,
}

##################################################################
######################## Model constructor #######################
##################################################################


def construct_model(name, summary = False, *args, **kwargs):
    if name in model_classes:
        custom_model = model_classes[name](*args, **kwargs)
        inputs = custom_model.base_model.input
        outputs = custom_model(inputs)
        if summary:
            custom_model.summary()
        return Model(inputs = inputs, outputs = outputs)
    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Model definition not found! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")