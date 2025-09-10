# utils.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import keras  # necesario para @keras.saving.register_keras_serializable


# --------------------------------------------------------------------------------------
# Utilidades de ROIs (no necesitan registro; se usan fuera del grafo)
# --------------------------------------------------------------------------------------
def getROIS(resolution=33, gridSize=3, minSize=1):
    coordsList = []
    step = resolution / gridSize
    for column1 in range(0, gridSize + 1):
        for column2 in range(0, gridSize + 1):
            for row1 in range(0, gridSize + 1):
                for row2 in range(0, gridSize + 1):
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)):
                        if not (x0 == y0 == 0 and x1 == y1 == resolution):  # ignora imagen completa
                            w = x1 - x0
                            h = y1 - y0
                            coordsList.append([x0, y0, w, h])
    return np.array(coordsList)


def getIntegralROIS(resolution=42, step=8, winSize=14):
    coordsList = []
    for column1 in range(0, resolution, step):
        for column2 in range(0, resolution, step):
            for row1 in range(column1 + winSize, resolution + winSize, winSize):
                for row2 in range(column2 + winSize, resolution + winSize, winSize):
                    if row1 > resolution or row2 > resolution:
                        continue
                    x0 = int(column1)
                    y0 = int(column2)
                    x1 = int(row1)
                    y1 = int(row2)
                    if not (x0 == y0 == 0 and x1 == y1 == resolution):  # ignora imagen completa
                        w = x1 - x0
                        h = y1 - y0
                        coordsList.append([x0, y0, w, h])
    return np.array(coordsList)


# --------------------------------------------------------------------------------------
# Funciones usadas en Lambda: ¡regístralas para que sean serializables!
# --------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="i2hofi")
def squeezefunc(x):
    return K.squeeze(x, axis=1)


@keras.saving.register_keras_serializable(package="i2hofi")
def stackfunc(x):
    # x será una lista de tensores si la Lambda recibe múltiples entradas
    return K.stack(x, axis=1)


# --------------------------------------------------------------------------------------
# Capa de "crop" serializable (evita closures no serializables)
# --------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="i2hofi")
class CropLayer(layers.Layer):
    """
    Recorta (slice) el tensor en la dimensión indicada: equivalente a usar Lambda con slices.
    dimension ∈ {0,1,2,3,4} y start/end son índices enteros.
    """
    def __init__(self, dimension, start, end, **kwargs):
        super().__init__(**kwargs)
        self.dimension = int(dimension)
        self.start = int(start)
        self.end = int(end)

    def call(self, x):
        if self.dimension == 0:
            return x[self.start:self.end]
        if self.dimension == 1:
            return x[:, self.start:self.end]
        if self.dimension == 2:
            return x[:, :, self.start:self.end]
        if self.dimension == 3:
            return x[:, :, :, self.start:self.end]
        if self.dimension == 4:
            return x[:, :, :, :, self.start:self.end]
        # por si acaso
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "dimension": self.dimension,
            "start": self.start,
            "end": self.end,
        })
        return cfg


def crop(dimension, start, end):
    """
    Factory para mantener la misma API que usas en models.py:
      crop(1, j, j+1)(tensor)
    Devuelve una instancia de CropLayer serializable.
    """
    return CropLayer(dimension, start, end)


# --------------------------------------------------------------------------------------
# ROI Pooling serializable
# --------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="i2hofi")
class RoiPoolingConv(Layer):
    """
    ROI pooling 2D (orden de ejes 'tf'): produce (B, num_rois, pool, pool, C)
    Usa rois precomputados (x,y,w,h) en pixeles sobre el feature map de entrada.
    """
    def __init__(self, pool_size, num_rois, rois_mat, **kwargs):
        super().__init__(**kwargs)
        self.dim_ordering = "tf"
        assert self.dim_ordering in {'tf', 'th'}, "dim_ordering must be 'tf' or 'th'"
        self.pool_size = int(pool_size)
        self.num_rois = int(num_rois)
        # Guardamos como np.ndarray; convertiremos a list en get_config
        self.rois = np.array(rois_mat, dtype=np.int32)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        else:
            self.nb_channels = input_shape[3]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return (None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size)
        else:
            return (None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)

    def call(self, img, mask=None):
        input_shape = K.shape(img)
        outputs = []

        for roi_idx in range(self.num_rois):
            x = self.rois[roi_idx, 0]
            y = self.rois[roi_idx, 1]
            w = self.rois[roi_idx, 2]
            h = self.rois[roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize(
                img[:, y:y + h, x:x + w, :],
                (self.pool_size, self.pool_size)
            )
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        base = super().get_config()
        cfg = {
            "pool_size": self.pool_size,
            "num_rois": self.num_rois,
            # convertir a list para que sea JSON-serializable
            "rois_mat": self.rois.tolist(),
        }
        base.update(cfg)
        return base

    @classmethod
    def from_config(cls, config):
        # rois_mat vuelve a np.array
        rois_mat = np.array(config.pop("rois_mat"), dtype=np.int32)
        return cls(rois_mat=rois_mat, **config)


# --------------------------------------------------------------------------------------
# FLOPs (opcional – igual que tu versión)
# --------------------------------------------------------------------------------------
def get_flops(model=None, in_tensor=None):
    """
    Calcula FLOPs usando TF v1 profiler.
    model: callable de Keras (modelo ya construido que acepta un tensor).
    in_tensor: tf.compat.v1.placeholder con la forma deseada.
    """
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            # Ejecuta el modelo sobre el placeholder para poblar el grafo
            _ = model(in_tensor)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops
