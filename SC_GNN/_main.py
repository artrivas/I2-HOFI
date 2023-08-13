# Importing necessary packages and modules
import sys
import os, ast
import yaml, json
import numpy as np

from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, GlobalAttentionPool, SortPool, TopKPool, GlobalSumPool, GlobalAttnSumPool, ARMAConv, APPNPConv
# from SelfAttention import SelfAttention

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM
from tensorflow.keras import layers 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as pp_input
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop


# user-defined functions (from utils.py)
from RoiPoolingConvTF2 import RoiPoolingConv
from opt_dg_tf2_new import DirectoryDataGenerator
from custom_validate_callback import CustomCallback
from utils import getROIS, getIntegralROIS, crop, squeezefunc, stackfunc
from schedulers import StepLearningRateScheduler

# from models import SR_GNN as model
from models import construct_model


tf.compat.v1.experimental.output_all_intermediates(True)


################################################# fx ##########################################################

"""  Function for pre-processing directory information """
def process_dir(dataset, model_name):

    dataset_dir = './datasets/' + dataset

    working_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_dir = '{}/train/'.format(dataset_dir)
    val_data_dir = '{}/val/'.format(dataset_dir)
    if not os.path.isdir(val_data_dir):
        val_data_dir = '{}/test/'.format(dataset_dir)

    output_model_dir = '{}/TrainedModels/{}'.format(working_dir, model_name)
    metrics_dir = '{}/Metrics/{}'.format(working_dir, model_name)

    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)]) # number of images used for training, including "other" action 
    nb_test_samples = 0 # number of images used for testing
    nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)]) # number of images used for validation
    validation_steps = validation_freq

    return dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps


""" Function for assigning varibales from config.yaml file """
all_vars = []
def assign_variables(dictionary, prefix=''):
    for key, value in dictionary.items():
        variable_name = key
        if isinstance(value, dict):
            assign_variables(value, variable_name + '_')
        else:
            globals()[variable_name] = value
            all_vars.append(variable_name)

###################################################################################################################

"""  Load and assign variables from the config file   """
param_dir = "./config.yaml"
with open(param_dir, 'r') as file:
    param = yaml.load(file, Loader = yaml.FullLoader)
print('Loading Default parameter configuration: \n', json.dumps(param, sort_keys = True, indent = 3))

# Assign 
assign_variables(param)


"""  Check and override with console paramaters  """
if len(sys.argv) > 2: # param 1 is file name
    total_params = len(sys.argv)
    for i in range(1, total_params, 2):
        var_name = sys.argv[i]
        new_val = sys.argv[i+1]
        try:
            exec("{} = {}".format(var_name, new_val))
        except:
            exec("{} = '{}'".format(var_name, new_val))

print(' ')
print('~~~~~~~~~~ << After updating with user parameters from console >> ~~~~~~~~~~~~~ ')
for v in all_vars:
    print(v, ':', globals()[v])
print(' ')


"""  << Additonal parameters and device settings >>  """
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.2)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
tf.compat.v1.disable_eager_execution()


"""  << Fetcing directory info >>  """ 
dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps = process_dir(dataset, model_name)
print('\n Dataset Location --> ', dataset_dir, '\n', 'no_of_class --> ', nb_classes, '\n', 'input_image_size ---> ', image_size, '\n', 'model_name -->', model_name)
print(train_data_dir, val_data_dir)
###########################################################################################################

# ============ Building model ============== #
model = construct_model(
    name = model_name,
    pool_size = pool_size,
    ROIS_resolution = ROIS_resolution,
    ROIS_grid_size = ROIS_grid_size,
    minSize = minSize,
    alpha = alpha,
    nb_classes = nb_classes,
    batch_size = batch_size,
    gcn_outfeat_dim = gcn_outfeat_dim,
    gat_outfeat_dim = gat_outfeat_dim,
    dropout_rate = dropout_rate,
    l2_reg = l2_reg,
    attn_heads = attn_heads,
    appnp_activation = appnp_activation,
    gat_activation = gat_activation,
    concat_heads = concat_heads,
    summary = summary
    )


# ~~~~~~~~~~~~~~~~~~~     Building log file      ~~~~~~~~~~~~~~~~~~~
checkpointer = ModelCheckpoint(
    filepath = output_model_dir + '.{epoch:02d}.h5', 
    verbose = 1, 
    save_weights_only = False, 
    period = checkpoint_freq
    )


train_dg = DirectoryDataGenerator(
    base_directories=[train_data_dir], 
    augmentor=True, 
    target_sizes=image_size, 
    preprocessors=pp_input, 
    batch_size=batch_size, 
    shuffle=True, 
    channel_last=True, 
    verbose=1, 
    hasROIS=False
    )

val_dg = DirectoryDataGenerator(
    base_directories=[val_data_dir], 
    augmentor=None, 
    target_sizes=image_size, 
    preprocessors=pp_input, 
    batch_size=batch_size, 
    shuffle=False, 
    channel_last=True, 
    verbose=1, 
    hasROIS=False
    )


steps_per_epoch = nb_train_samples // batch_size
callbacks = [checkpointer, CustomCallback(val_dg, validation_steps, metrics_dir + model_name, wandb_log)]

print('<<<<<<<<<<<<<<<<<<<<< Learning_rate reduction used >>>>>>>>>>>>>>>>>>>>>>>>  ', reduce_lr_bool)
if reduce_lr_bool == True:
    # Define the epochs at which to reduce the learning rate and create lr scheduler callback
    schedule_epochs = [50, 100, 150]
    lr_schedule = StepLearningRateScheduler(schedule_epochs, factor = 0.1)
    # lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, min_lr = 1e-6, verbose = 1)
    callbacks.append(lr_schedule)


# ============  Building engine ============= #
print('++++++++++++++++++++++++++++++++ checkpoint path:', checkpoint_path)
# model.load_weights(checkpoint_path)
optimizer = SGD(learning_rate = lr) 
model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

# model.summary() # !!  print model summary (optional)

# ################ Training Model ############################

print('----> Steps_per_epoch :', steps_per_epoch)
model.fit(
    train_dg, 
    steps_per_epoch = steps_per_epoch, 
    initial_epoch = completed_epochs,  
    epochs = epochs, 
    callbacks = callbacks
    ) #train and validate the model