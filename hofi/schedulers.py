import tensorflow as tf
from tensorflow import keras

class StepLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule_epochs, factor=0.1, verbose=1):
        super().__init__()
        self.schedule_epochs = set(schedule_epochs)
        self.factor = factor
        self.verbose = verbose

    def _get_lr_var(self):
        opt = self.model.optimizer
        # Desencapsular si está envuelto (p.ej., LossScaleOptimizer)
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer
        lr_var = getattr(opt, "learning_rate", None)
        if lr_var is None:
            lr_var = getattr(opt, "lr", None)  # fallback legacy
        if lr_var is None:
            raise AttributeError("El optimizador no tiene 'learning_rate' ni 'lr'.")
        return lr_var

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.schedule_epochs:
            lr_var = self._get_lr_var()
            current = tf.keras.backend.get_value(lr_var)
            new = current * self.factor
            tf.keras.backend.set_value(lr_var, new)
            if self.verbose:
                print(f"\nEpoch {epoch:03d}: reducing LR from {current:.6g} to {new:.6g}")
