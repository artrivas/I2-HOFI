# -*- coding: utf-8 -*-
import sys
import os, ast
import yaml, json
import math
import numpy as np
import wandb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.applications.xception import preprocess_input as pp_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# user-defined functions (from utils.py)
from dataset_info import datasetInfo
from datagen import DirectoryDataGenerator
from customcallbacks import ValCallback
from schedulers import StepLearningRateScheduler  # opcional: ya no se usa por defecto
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
    train_data_dir = f'{dataset_dir}/train/'
    val_data_dir = f'{dataset_dir}/val/'
    if not os.path.isdir(val_data_dir):
        val_data_dir = f'{dataset_dir}/test/'

    # Carpeta de modelos y métricas por modelo
    output_model_dir = f'{working_dir}/TrainedModels/{model_name}'
    metrics_dir = f'{working_dir}/Metrics/{model_name}'

    # Crear carpetas si no existen
    ensure_dir(output_model_dir)
    ensure_dir(metrics_dir)

    # Conteos de imágenes
    nb_train_samples = sum([len(files) for _, _, files in os.walk(train_data_dir)])
    nb_val_samples = sum([len(files) for _, _, files in os.walk(val_data_dir)])

    return (dataset_dir, train_data_dir, val_data_dir,
            output_model_dir, metrics_dir, nb_train_samples, nb_val_samples)

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
    (dataset_dir, train_data_dir, val_data_dir,
     output_model_dir, metrics_dir, nb_train_samples, nb_val_samples) = process_dir(
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
                "validation_freq": validation_freq,
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
    # Cambiamos a formato Keras nativo para evitar warnings
    filename_template = (
        f"{dataset}_{backbone}_Bs{batch_size}_initlr{lr}_"
        "epoch:{:03d}_lr{:.6f}_valAcc{:.4f}.keras"
    )
    checkpoint_path = os.path.join(output_model_dir, filename_template)

    # =========== Custom CALLBACKS (val + guardado) ==========
    # IMPORTANTE: ValCallback debe guardar con model.save(...) si usas .keras
    # (si guarda solo pesos usa .weights.h5 y model.save_weights(...))
    # Ahora pasamos pasos de validación reales, NO validation_freq.
    # Calculamos steps por tamaño de dataset para estabilidad.
    train_steps = max(1, math.ceil(nb_train_samples / batch_size))
    val_steps   = max(1, math.ceil(nb_val_samples   / batch_size))

    callbacks.append(
        ValCallback(
            val_dg,
            val_steps,            # <- nº de batches a evaluar en validación (completo)
            metrics_dir,
            wandb_log,
            save_model,
            checkpoint_path,
            save_best_only,
            checkpoint_freq
        )
    )

    # =========== Scheduler/ES ===========
    # Por defecto: ReduceLROnPlateau + EarlyStopping para estabilizar val
    if reduce_lr_bool:
        callbacks.append(
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                              verbose=1, min_lr=1e-6)
        )

    callbacks.append(
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    )

    # (Opcional) Si quieres además tu Step LR, actívalo moviendo los hitos:
    # schedule_epochs = [10, 20, 30]
    # callbacks.append(StepLearningRateScheduler(schedule_epochs, factor=0.1))

    # =======================  Cargar pesos si corresponde  =======================
    if load_model:
        print('_____________ Checkpoint path to Load Pretrained Model _____________ :', checkpoint_path)
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            print("No se pudo cargar el checkpoint especificado:", e)

    # =======================  Compilar y Entrenar  =======================
    # Sugerido para Adam: lr ~ 1e-3; usamos clipnorm para estabilizar GATs
    # Si tu YAML trae lr alto (p. ej. 0.01), cámbialo a 0.001.
    optimizer = Adam(learning_rate=lr, amsgrad=True, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", TopKCategoricalAccuracy(k=5)],
        jit_compile=False,       # dejar False para evitar sorpresas con custom layers
        # run_eagerly=True,      # activar solo si necesitas depurar (lento)
    )

    history = model.fit(
        train_dg,
        steps_per_epoch=train_steps,
        validation_data=val_dg,
        validation_steps=val_steps,     # <- evaluamos TODO val
        validation_freq=validation_freq,  # <- cada cuántas épocas validar (del YAML)
        initial_epoch=completed_epochs,
        epochs=epochs,
        callbacks=callbacks
    )

    # =======================  Guardado final del modelo completo  =======================
    final_model_path = os.path.join(output_model_dir, f"{model_name}_final.keras")
    try:
        model.save(final_model_path)
        print("Modelo completo guardado en:", final_model_path)
    except Exception as e:
        print("Advertencia: no se pudo guardar el modelo completo (.keras):", e)

    # =======================  Cierre de W&B  =======================
    if wandb_log and (wrun is not None):
        wrun.finish()
