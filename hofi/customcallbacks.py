# -*- coding: utf-8 -*-
"""
@author: sikdara

from custom_validate_callback import ValCallback
"""
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from sklearn.metrics import accuracy_score  # si no lo usas, puedes eliminarlo
import wandb


def _current_lr(optimizer):
    """Obtiene el LR actual de forma robusta (float), soporte para schedules y variables."""
    lr = getattr(optimizer, "learning_rate", None)
    if lr is None:
        lr = getattr(optimizer, "lr", None)  # compat legacy
    if lr is None:
        return None

    # Si es un schedule (callable), evalúa en el step actual
    try:
        if callable(lr):
            return float(K.get_value(lr(optimizer.iterations)))
    except Exception:
        pass

    # Si es Variable/Tensor/float
    try:
        return float(K.get_value(lr))
    except Exception:
        try:
            return float(lr)
        except Exception:
            return None


class ValCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name,
                 wandb_log=False, save_model=False, checkpoint_path=None,
                 best_only=False, checkpoint_freq=1):
        self.test_generator = test_generator
        self.test_steps = test_steps              # frecuencia (en épocas) para validar/guardar
        self.model_name = model_name
        self.wandb_log = wandb_log
        self.model_save = save_model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.model_path = checkpoint_path
        self.best_only = best_only
        self.val_acc = 0.0
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # LR robusto
        lr = _current_lr(self.model.optimizer)
        lr_print = "N/A" if lr is None else f"{lr:.8f}"
        print(" - lr :", lr_print)

        # Métrica de entrenamiento: intenta 'accuracy', luego 'acc', luego otras
        train_acc = (logs.get("accuracy") or logs.get("acc")
                     or logs.get("categorical_accuracy") or logs.get("sparse_categorical_accuracy"))

        # Log a W&B (si procede)
        if self.wandb_log:
            data = {"epoch": epoch, "loss": logs.get("loss")}
            if train_acc is not None:
                data["acc"] = float(train_acc)
            if lr is not None:
                data["lr"] = float(lr)
            wandb.log(data)

        # Validación/guardado cada N épocas (y no en la 0)
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            # evaluar en el generador de validación
            loss, acc = self.model.evaluate(self.test_generator, verbose=0)

            if self.model_save and self.checkpoint_path:
                # ¿guardar mejor modelo o por frecuencia?
                do_save = False
                if self.best_only:
                    if acc > self.best_val_acc and self.val_acc > 0.0:
                        # borrar el archivo previo si existe
                        if self.model_path and os.path.exists(self.model_path):
                            os.remove(self.model_path)
                        do_save = True
                else:
                    if (epoch + 1) % self.checkpoint_freq == 0 and epoch != 0:
                        do_save = True

                if do_save:
                    # Asegura LR numérico en el nombre
                    lr_for_name = 0.0 if lr is None else float(lr)
                    self.model_path = self.checkpoint_path.format(epoch, lr_for_name, acc)
                    self.model.save(self.model_path)

            self.val_acc = float(acc)
            self.best_val_acc = max(self.best_val_acc, self.val_acc)

            if self.wandb_log:
                wandb.log({"val_loss": float(loss), "val_acc": float(acc)})


