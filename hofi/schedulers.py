import tensorflow as tf
from tensorflow import keras

class StepLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule_epochs, factor=0.1, verbose=1):
        super().__init__()
        self.schedule_epochs = set(schedule_epochs)
        self.factor = factor
        self.verbose = verbose

    def _unwrap_opt(self, opt):
        # Desencapsula (p. ej., LossScaleOptimizer)
        while hasattr(opt, "optimizer"):
            opt = opt.optimizer
        return opt

    def _get_lr_obj_and_attr(self):
        opt = self._unwrap_opt(self.model.optimizer)
        if hasattr(opt, "learning_rate"):
            return opt, "learning_rate", getattr(opt, "learning_rate")
        if hasattr(opt, "lr"):  # legacy
            return opt, "lr", getattr(opt, "lr")
        raise AttributeError("El optimizador no tiene 'learning_rate' ni 'lr'.")

    def _read_lr(self, lr_obj):
        # tf.Variable
        if isinstance(lr_obj, tf.Variable):
            return float(lr_obj.numpy())
        # Keras/TF schedules: intenta leer el valor base
        if hasattr(lr_obj, "initial_learning_rate"):
            base = lr_obj.initial_learning_rate
            try:
                return float(base.numpy() if hasattr(base, "numpy") else base)
            except Exception:
                pass
        # Números o strings
        try:
            return float(lr_obj)
        except Exception:
            raise TypeError(
                "No se pudo leer el learning rate actual. ¿Usas un schedule no mutable?"
            )

    def _write_lr(self, opt, attr_name, lr_obj, new_value):
        # tf.Variable → assign
        if isinstance(lr_obj, tf.Variable):
            lr_obj.assign(new_value)
            return
        # Schedule con 'initial_learning_rate' → actualiza el base LR
        if hasattr(lr_obj, "initial_learning_rate"):
            lr_obj.initial_learning_rate = new_value
            return
        # float/int/str → asignación directa al atributo del optimizador
        setattr(opt, attr_name, float(new_value))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.schedule_epochs:
            opt, attr_name, lr_obj = self._get_lr_obj_and_attr()
            current = self._read_lr(lr_obj)
            new = current * self.factor
            self._write_lr(opt, attr_name, lr_obj, new)
            if self.verbose:
                print(f"\nEpoch {epoch:03d}: reducing LR from {current:.6g} to {new:.6g}")

