from tensorflow.keras.models import Model
from .models import *

""" Importing model definations """
model_classes = {
    "basecnn": BASE_CNN,
    "i2hofi": I2HOFI,
}

##################################################################
###################### Model constructor #########################
##################################################################

def construct_model(name, *args, **kwargs):
    if name in model_classes:
        # Build and return model with user arguments
        return model_classes[name](*args, **kwargs)
    else:
        print("@@@@@@@@@@@@@@@@ Unknown Model definition! @@@@@@@@@@@@@@@@@@@@@")
