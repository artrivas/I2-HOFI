
from tensorflow.keras.models import Model
# from .models0 import *
from .models import *


""" Importing model definations """
model_classes = {
    # Base models
    "basecnn": BASE_CNN,
    "scgnn": SC_GNN,
    
    # Add more models
    "scgnn_intra": SC_GNN_INTRA,
    "scgnn_inter": SC_GNN_INTER,
    "scgnn_comb": SC_GNN_COMB,
    
    # Models with residual connection
    "scgnnres_intra": SC_GNNRES_INTRA,
    "scgnnres_inter": SC_GNNRES_INTER,
    "scgnnres_comb": SC_GNNRES_COMB,

    # Models with Residual + GATconv
    "scgnngatres_intra": SC_APPNPGATRES_INTRA,
    "scgnngatres_inter": SC_APPNPGATRES_INTER,
    "scgnngatres_comb": SC_APPNPGATRES_COMB,

    # # Model for t-sne outputs
    # "scgnngatres_comb_vis": SC_APPNPGATRES_COMB_VIS,
}

##################################################################
######################## Model constructor #######################
##################################################################


# def construct_model(name, summary = False, *args, **kwargs):
#     if name in model_classes:
#         custom_model = model_classes[name](*args, **kwargs)
#         inputs = custom_model.base_model.input
#         outputs = custom_model(inputs)
#         if summary:
#             custom_model.summary()
#         return Model(inputs = inputs, outputs = outputs)
#     else:
#         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Model definition not found! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



def construct_model(name, *args, **kwargs):
    if name in model_classes:
        custom_model = model_classes[name](*args, **kwargs)

        # Return the custom model without creating a single-layer model
        return custom_model
    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Model definition not found! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")