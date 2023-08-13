

import tensorflow as tf

# ######### Learning rate schedular class 
class StepLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule_epochs, factor=0.1):
        super(StepLearningRateScheduler, self).__init__()
        self.schedule_epochs = set(schedule_epochs)
        self.factor = factor

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.schedule_epochs:
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = lr * self.factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nEpoch {epoch + 1}: Learning rate reduced to {new_lr}")