# -*- coding: utf-8 -*-
import sys
import os, ast
import yaml, json
import numpy as np
import wandb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.xception import preprocess_input as pp_input

# user-defined functions (from utils.py)
from dataset_info import datasetInfo
from datagen import DirectoryDataGenerator
from customcallbacks import ValCallback
from schedulers import StepLearningRateScheduler
from utils import get_flops
from models import construct_model

# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    """Crea la carpeta si no existe."""
    os.makedirs(path, exist_ok=True)

# ######################################### PROCESSING DATASET DIRECTORY INFO ####################################### #
"""  Function for pre-processing directory information """
def process_dir(rootdir, dataset, model_name):
    dataset_dir = rootdir + dataset
    working_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_dir = '{}/train/'.format(dataset_dir)
    val_data_dir = '{}/val/'.format(dataset_dir)
    if not os.path.isdir(val_data_dir):
        val_data_dir = '{}/test/'.format(dataset_dir)

    # Carpeta de modelos y métricas por modelo
    output_model_dir = '{}/TrainedModels/{}'.format(working_dir, model_name)
    metrics_dir = '{}/Metrics/{}'.format(working_dir, model_name)

    # Crear carpetas si no existen
    ensure_dir(output_model_dir)
    ensure_dir(metrics_dir)

    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])  # número de imágenes de train
    nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)])      # número de imágenes de val/test

    # validation_steps viene desde el config como validation_freq
    validation_steps = validation_freq

    return dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps

# ======================= Cargar config y variables =======================

""" Function for assigning variables from config.yaml file """
all_vars = []
def assign_variables(dictionary, prefix=''):
    for key, value in dictionary.items():
        variable_name = key
        if isinstance(value, dict):
            assign_variables(value, variable_name + '_')
        else:
            globals()[variable_name] = value
            all_vars.append(variable_name)

if __name__ == "__main__":
    # --------------------- Cargar config ---------------------
    dataset_name = sys.argv[sys.argv.index('dataset') + 1] if 'dataset' in sys.argv else None
    try:
        param_dir = "./configs/config_" + dataset_name + ".yaml"
    except Exception:
        print('Please provide a valid dataset name under the dataset argument; Example command ---> python hofi/train.py dataset Aircraft')
        sys.exit(1)

    with open(param_dir, 'r') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    print('Loading Default parameter configuration: \n', json.dumps(param, sort_keys=True, indent=3))

    # Asignar variables del YAML a globals()
    assign_variables(param)

    # ----------------- Override por CLI opcional -----------------
    if len(sys.argv) > 2:
        total_params = len(sys.argv)
        for i in range(1, total_params, 2):
            var_name = sys.argv[i]
            new_val = sys.argv[i + 1]
            try:
                exec("{} = {}".format(var_name, new_val))
            except Exception:
                exec("{} = '{}'".format(var_name, new_val))

    # ----------------- Paths y conteos -----------------
    dataset_dir, train_data_dir, val_data_dir, output_model_dir, metrics_dir, nb_train_samples, validation_steps = process_dir(
        rootdir, dataset, model_name
    )
    nb_classes = datasetInfo(dataset)
    print('\n Dataset Location --> ', dataset_dir, '\n', 'no_of_class --> ', nb_classes)
    print('_________________ Wandb_Logging _______________ : ', wandb_log)
    print(' ')

    # ----------------- Dispositivo -----------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # -1 -> CPU, 0/1 -> GPU
    physical = tf.config.list_physical_devices("GPU")
    if physical:
        try:
            tf.config.experimental.set_memory_growth(physical[0], True)
        except Exception as e:
            print("Warning: set_memory_growth failed:", e)

    # ===================== W&B ======================
    wrun = None
    if wandb_log:
        wandb.login(key=API_key)  # WandB API key
        wrun = wandb.init(
            project=wb_proj_name,
            name=model_name if run_name == 'None' else run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "nb_classes": nb_classes,
                "lr": lr,
                "validation_steps": validation_steps,
                "checkpoint_freq": checkpoint_freq,
                "completed_epochs": completed_epochs,
                "gcn_outfeat_dim": gcn_outfeat_dim,
                "gat_outfeat_dim": gat_outfeat_dim,
                "dropout_rate": dropout_rate,
                "l2_reg": l2_reg,
                "attn_heads": attn_heads,
                "appnp_activation": appnp_activation,
                "gat_activation": gat_activation,
                "concat_heads": concat_heads,
                "reduce_lr_bool": reduce_lr_bool,
                "backbone": backbone,
                "freeze_backbone": freeze_backbone,
                "GNN_layer1": gnn1_layr,
                "GNN_layer2": gnn2_layr,
                "alpha": alpha,
                "pool_size": pool_size,
                "input_sh": image_size,
            }
        )

    # ================== Construir modelo base ==================
    model = construct_model(
        name=model_name,
        pool_size=pool_size,
        ROIS_resolution=ROIS_resolution,
        ROIS_grid_size=ROIS_grid_size,
        minSize=minSize,
        alpha=alpha,
        nb_classes=nb_classes,
        batch_size=batch_size,
        input_sh=image_size,
        gcn_outfeat_dim=gcn_outfeat_dim,
        gat_outfeat_dim=gat_outfeat_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        attn_heads=attn_heads,
        appnp_activation=appnp_activation,
        gat_activation=gat_activation,
        concat_heads=concat_heads,
        backbone=backbone,                # 'Xception' (sensible a mayúsculas)
        freeze_backbone=freeze_backbone,  # congelar backbone si se desea
        gnn1_layr=gnn1_layr,
        gnn2_layr=gnn2_layr,
        track_feat=track_feat,
    )

    # ======= Cálculo de FLOPs opcional =======
    if calflops:
        r = get_flops(model, tf.compat.v1.placeholder('float32', shape=(1, image_size[0], image_size[1], image_size[2])))
        print('~~~~~~~~ Total FLOPs --> {} | Giga-FLOPs --> {} ~~~~~~~~'.format(r, r / 10 ** 9))
        if summary:
            model.summary()
        sys.exit(0)

    # ======= Inicializar grafo funcional =======
    outputs = model(model.base_model.input)
    if summary:
        model.summary()
    model = Model(inputs=model.input, outputs=outputs)

    # ================== Data Generators ==================
    callbacks = []

    train_dg = DirectoryDataGenerator(
        base_directories=[train_data_dir],
        augmentor=True,
        target_sizes=image_size[:2],
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
        target_sizes=image_size[:2],
        preprocessors=pp_input,
        batch_size=batch_size,
        shuffle=False,
        channel_last=True,
        verbose=1,
        hasROIS=False
    )

    # ================== Checkpoint Path ==================
    # Usamos separadores seguros (sin '|') y mantenemos placeholders para el ValCallback
    filename_template = (
        f"{dataset}_{backbone}_Bs{batch_size}_initlr{lr}_"
        "epoch:{:03d}_lr{:.6f}_valAcc{:.4f}.h5"
    )
    checkpoint_path = os.path.join(output_model_dir, filename_template)

    # =========== Custom CALLBACKS (val + guardado) ==========
    callbacks.append(
        ValCallback(
            val_dg,
            validation_steps,
            metrics_dir,          # <- SOLO metrics_dir (sin sumar model_name)
            wandb_log,
            save_model,
            checkpoint_path,
            save_best_only,
            checkpoint_freq
        )
    )

    # =========== LR Scheduler opcional ===========
    if reduce_lr_bool:
        schedule_epochs = [50, 100, 150]
        lr_schedule = StepLearningRateScheduler(schedule_epochs, factor=0.1)
        callbacks.append(lr_schedule)

    # =======================  Cargar pesos si corresponde  =======================
    if load_model:
        print('_____________ Checkpoint path to Load Pretrained Model _____________ :', checkpoint_path)
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            print("No se pudo cargar el checkpoint especificado:", e)

    # =======================  Compilar y Entrenar  =======================
    optimizer = SGD(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,       # dejar False para evitar sorpresas con custom layers
        # run_eagerly=True,      # activar solo si necesitas depurar (lento)
    )

    steps_per_epoch = max(1, nb_train_samples // batch_size)  # evitar 0
    history = model.fit(
        train_dg,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dg,
        validation_steps=validation_steps,
        initial_epoch=completed_epochs,
        epochs=epochs,
        callbacks=callbacks
    )

    # =======================  Guardado final del modelo completo  =======================
    # Además de los checkpoints de pesos, guarda un .keras (modelo completo listo para cargar)
    final_model_path = os.path.join(output_model_dir, f"{model_name}_final.keras")
    try:
        model.save(final_model_path)
        print("Modelo completo guardado en:", final_model_path)
    except Exception as e:
        print("Advertencia: no se pudo guardar el modelo completo (.keras):", e)

    # =======================  Cierre de W&B  =======================
    if wandb_log and (wrun is not None):
        wrun.finish()

