# models.py
# -*- coding: utf-8 -*-
"""
Modelos serializables para Keras 3 (CNN + GNN).
Requiere:
- spektral
- utils.py con: RoiPoolingConv, getROIS, crop, squeezefunc, stackfunc (registrados)
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
import keras  # <- necesario para @keras.saving.register_keras_serializable

# Spektral
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, GlobalAttentionPool
from spektral.layers import GATConv as _GATConv
from spektral.layers import APPNPConv as _APPNPConv

# Utils propios (asegúrate de registrar sus símbolos en utils.py)
from utils import RoiPoolingConv, getROIS, crop, squeezefunc, stackfunc


# ------------------------------------------------------------------------------
# Capas/Clases auxiliares registradas
# ------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="i2hofi")
class APPNPConvSafe(_APPNPConv):
    """APPNP que tolera mask=[None, None] (Keras 3)"""
    def call(self, inputs, mask=None, **kwargs):
        if isinstance(mask, (list, tuple)) and all(m is None for m in mask):
            mask = None
        return super().call(inputs, mask=mask, **kwargs)

    def get_config(self):
        cfg = super().get_config()
        return cfg


@keras.saving.register_keras_serializable(package="i2hofi")
class GATConvPatched(_GATConv):
    """GAT que tolera mask=[None, None] y evita K.eval en camino denso."""
    def call(self, inputs, mask=None, **kwargs):
        if isinstance(mask, (list, tuple)) and all(m is None for m in mask):
            mask = None
        return super().call(inputs, mask=mask, **kwargs)

    def _call_dense(self, x, a):
        a_dense = tf.sparse.to_dense(a) if isinstance(a, tf.SparseTensor) else a

        if self.add_self_loops:
            n = tf.shape(a_dense)[-1]
            a_dense = tf.linalg.set_diag(a_dense, tf.ones([n], dtype=a_dense.dtype))

        # Proyecciones
        x = tf.einsum("...NI, IHO -> ...NHO", x, self.kernel)
        attn_self  = tf.einsum("...NHI, IHO -> ...NHO", x, self.attn_kernel_self)
        attn_neigh = tf.einsum("...NHI, IHO -> ...NHO", x, self.attn_kernel_neighs)
        attn_neigh = tf.einsum("...ABC -> ...CBA", attn_neigh)

        attn_coef = tf.nn.leaky_relu(attn_self + attn_neigh, alpha=0.2)

        minus_inf = tf.cast(-1e9, attn_coef.dtype)
        mask_logits = tf.where(tf.equal(a_dense, 0.0), minus_inf, 0.0)
        attn_coef = attn_coef + mask_logits[..., None, :]

        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)
        out = tf.einsum("...NHM, ...MHI -> ...NHI", attn_coef_drop, x)
        return out, attn_coef

    def get_config(self):
        cfg = super().get_config()
        return cfg


@keras.saving.register_keras_serializable(package="i2hofi")
class TempNodesTransform(layers.Layer):
    """
    Convierte cada ROI [p, p, C] en una secuencia de nodos concatenando
    splits de canales en bloques de tamaño gcn_outfeat_dim:
      r x p x p x C -> r x (p*p*C/gcn_outfeat_dim) x gcn_outfeat_dim
    """
    def __init__(self, pool_size, base_channels, gcn_outfeat_dim, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = int(pool_size)
        self.base_channels = int(base_channels)
        self.gcn_outfeat_dim = int(gcn_outfeat_dim)

    def call(self, roi):
        reshaped = tf.reshape(roi, (-1, self.pool_size * self.pool_size, self.base_channels))
        splits = tf.split(reshaped, num_or_size_splits=self.base_channels // self.gcn_outfeat_dim, axis=2)
        joined = tf.concat(splits, axis=1)
        return joined

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "pool_size": self.pool_size,
            "base_channels": self.base_channels,
            "gcn_outfeat_dim": self.gcn_outfeat_dim,
        })
        return cfg


# ------------------------------------------------------------------------------
# Clase base de parámetros/estructura
# ------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="i2hofi")
class Params(Model):
    """
    Inicializa parámetros y backbone CNN para modelos CNN+GNN.
    """
    def __init__(
        self,
        pool_size=None,
        ROIS_resolution=None,
        ROIS_grid_size=None,
        minSize=None,
        alpha=None,
        nb_classes=None,
        batch_size=None,
        input_sh=(224, 224, 3),
        gcn_outfeat_dim=256,
        gat_outfeat_dim=256,
        dropout_rate=0.2,
        l2_reg=2.5e-4,
        attn_heads=1,
        appnp_activation='sigmoid',
        gat_activation='elu',
        concat_heads=True,
        backbone=None,
        freeze_backbone=None,
        gnn1_layr=True,
        gnn2_layr=True,
        track_feat=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Guardar hiperparámetros para serialización
        self.pool_size = pool_size
        self.ROIS_resolution = ROIS_resolution
        self.ROIS_grid_size = ROIS_grid_size
        self.minSize = minSize
        self.alpha = alpha
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_sh = tuple(input_sh) if isinstance(input_sh, (list, tuple)) else input_sh
        self.gcn_outfeat_dim = int(gcn_outfeat_dim)
        self.gat_outfeat_dim = int(gat_outfeat_dim)
        self.dropout_rate = float(dropout_rate)
        self.l2_reg = float(l2_reg)
        self.attn_heads = int(attn_heads)
        self.appnp_activation = appnp_activation
        self.gat_activation = gat_activation
        self.concat_heads = bool(concat_heads)
        self.backbone_str = backbone
        self.freeze_backbone = bool(freeze_backbone) if freeze_backbone is not None else False
        self.gnn1_layr = bool(gnn1_layr)
        self.gnn2_layr = bool(gnn2_layr)
        self.track_feat = bool(track_feat)

        # Cargar backbone
        if self.backbone_str is None:
            raise ValueError("Debe especificar 'backbone' (p.ej., 'Xception').")
        base_model_class = getattr(tf.keras.applications, self.backbone_str)
        self.base_model = base_model_class(
            weights="imagenet",
            input_tensor=layers.Input(shape=self.input_sh),
            include_top=False,
        )

        # Congelar backbone si aplica
        if self.freeze_backbone:
            for layer in self.base_model.layers:
                layer.trainable = False

        # Ajuste de dimensión de salida de GAT por número de cabezas si se concatena
        if self.concat_heads and self.attn_heads > 0:
            self.gat_outfeat_dim = self.gat_outfeat_dim // self.attn_heads

    # ----- serialización -----
    def get_config(self):
        return {
            "pool_size": self.pool_size,
            "ROIS_resolution": self.ROIS_resolution,
            "ROIS_grid_size": self.ROIS_grid_size,
            "minSize": self.minSize,
            "alpha": self.alpha,
            "nb_classes": self.nb_classes,
            "batch_size": self.batch_size,
            "input_sh": self.input_sh,
            "gcn_outfeat_dim": self.gcn_outfeat_dim,
            "gat_outfeat_dim": self.gat_outfeat_dim,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "attn_heads": self.attn_heads,
            "appnp_activation": self.appnp_activation,
            "gat_activation": self.gat_activation,
            "concat_heads": self.concat_heads,
            "backbone": self.backbone_str,
            "freeze_backbone": self.freeze_backbone,
            "gnn1_layr": self.gnn1_layr,
            "gnn2_layr": self.gnn2_layr,
            "track_feat": self.track_feat,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ------------------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="i2hofi")
class BASE_CNN(Params):
    """
    Backbone CNN + GAP + Dense softmax (baseline).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._construct_layers()

    def _construct_layers(self):
        self.GAP_layer = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.nb_classes, activation="softmax")

    def call(self, inputs):
        base_out = self.base_model(inputs)
        x_gap = self.GAP_layer(base_out)
        x = self.dense(x_gap)

        if self.track_feat:
            self.base_out = tf.identity(base_out)
            self.GlobAttpool_feat = tf.identity(x_gap)

        return x

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="i2hofi")
class I2HOFI(Params):
    """
    Interweaving Insights: High-Order Feature Interaction for FGVR (CNN+GNN con APPNP + GAT).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dimensiones del backbone
        dims = tf.keras.backend.int_shape(self.base_model.output)[1:]
        if dims is None or len(dims) != 3:
            raise ValueError("No se pudo inferir (H, W, C) del backbone.")
        h, w, c = dims
        self.base_channels = int(c)
        self.feat_dim = int(self.base_channels) * int(self.pool_size) * int(self.pool_size)

        # ROIs
        self.rois_mat = getROIS(
            resolution=self.ROIS_resolution,
            gridSize=self.ROIS_grid_size,
            minSize=self.minSize,
        )
        self.num_rois = int(self.rois_mat.shape[0])

        # Número de nodos por ROI (tras reshape y splits de canales)
        self.cnodes = (self.base_channels // self.gcn_outfeat_dim) * self.pool_size * self.pool_size

        # Capas/adjacencias
        self._construct_adjecency()
        self._construct_layers()

    # ----- grafo -----
    def _construct_adjecency(self):
        # Intra-ROI
        A1 = np.ones((self.cnodes, self.cnodes), dtype="float32")
        self.A_intra = tf.sparse.reorder(
            sp_matrix_to_sp_tensor(GCNConv.preprocess(A1).astype("f4"))
        )

        # Inter-ROI
        A2 = np.ones((self.num_rois + 1, self.num_rois + 1), dtype="float32")
        self.A_inter = tf.sparse.reorder(
            sp_matrix_to_sp_tensor(GCNConv.preprocess(A2).astype("f4"))
        )

    def _extract_roi_nodes(self, x0, base_out):
        # ROI pooling
        roi_pool = self.roi_pooling(x0)

        jcvs = []
        for j in range(self.num_rois):
            roi_crop = crop(1, j, j + 1)(roi_pool)
            lname = "roi_lambda_" + str(j)
            x = layers.Lambda(squeezefunc, name=lname)(roi_crop)
            x = layers.Reshape((self.feat_dim,))(x)
            jcvs.append(x)

        # Añadir el mapa base como ROI adicional
        if self.pool_size != base_out.shape[1]:
            base_out = layers.Lambda(
                lambda x: tf.image.resize(x, size=(self.pool_size, self.pool_size)),
                name="Lambda_img_2",
            )(base_out)

        x = layers.Reshape((self.feat_dim,))(base_out)
        jcvs.append(x)

        # Apilar ROIs
        jcvs = layers.Lambda(stackfunc, name="lambda_stack")(jcvs)
        jcvs = self.roi_droput_1(jcvs)
        return jcvs

    def _construct_layers(self):
        # Upsampling del feature map del backbone
        self.upsampling_layer = layers.Lambda(
            lambda x: tf.image.resize(x, size=(self.ROIS_resolution, self.ROIS_resolution)),
            name="UpSample",
        )

        # ROI pooling
        self.roi_pooling = RoiPoolingConv(
            pool_size=self.pool_size, num_rois=self.num_rois, rois_mat=self.rois_mat
        )

        # Dropout tras ROI pooling
        self.roi_droput_1 = layers.Dropout(self.dropout_rate, name="DOUT_1")

        # TimeDistributed: reshape a [r, p, p, C]
        self.timedist_layer1 = layers.TimeDistributed(
            layers.Reshape((self.pool_size, self.pool_size, self.base_channels)),
            name="TD_Layer1",
        )

        # TimeDistributed: convertir a nodos (sin Lambda con método ligado)
        self.timedist_layer2 = layers.TimeDistributed(
            TempNodesTransform(self.pool_size, self.base_channels, self.gcn_outfeat_dim),
            name="TD_Layer2",
        )

        # GNN 1: APPNP
        if self.gnn1_layr:
            self.tgcn_1 = APPNPConvSafe(
                self.gcn_outfeat_dim,
                alpha=self.alpha,
                propagations=1,
                mlp_activation=self.appnp_activation,
                use_bias=True,
                name="GNN_1",
            )

        # GNN 2: GAT
        if self.gnn2_layr:
            self.tgcn_2 = GATConvPatched(
                self.gat_outfeat_dim,
                attn_heads=self.attn_heads,
                concat_heads=self.concat_heads,
                dropout_rate=self.dropout_rate,
                activation=self.gat_activation,
                kernel_regularizer=l2(self.l2_reg),
                attn_kernel_regularizer=l2(self.l2_reg),
                bias_regularizer=l2(self.l2_reg),
                name="GNN_2",
            )

        # Dropout tras combinar nodos
        self.roi_droput_2 = layers.Dropout(self.dropout_rate, name="DOUT_2")

        # Pooling global + BN + clasificador
        self.GlobAttpool = GlobalAttentionPool(self.gcn_outfeat_dim * 2, name="GlobalAttnPool")
        self.BN2 = layers.BatchNormalization(name="BN")
        self.Dense = layers.Dense(self.nb_classes, activation="softmax", name="Fully_Conn")

    def call(self, inputs):
        # Backbone
        base_out = self.base_model(inputs)
        if self.track_feat:
            self.base_out = tf.identity(base_out)

        # Upsample
        x0 = self.upsampling_layer(base_out)

        # ROIs
        rois = self._extract_roi_nodes(x0, base_out)

        # TD reshape + a nodos
        x1 = self.timedist_layer1(rois)
        x1 = self.timedist_layer2(x1)

        # Intra-ROI
        splits = tf.split(x1, num_or_size_splits=self.num_rois + 1, axis=1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.A_intra])
                x = temp + x
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.A_intra])
                temp = temp + x
            xcoll.append(temp)
        x2_intra = tf.concat(xcoll, axis=1)

        if self.track_feat:
            self.x2_intra = tf.identity(x2_intra)

        # Inter-ROI
        x1_t = tf.transpose(x1, perm=[0, 2, 1, 3])  # [B, cnodes, r, feat]
        splits = tf.split(x1_t, num_or_size_splits=self.cnodes, axis=1)
        xcoll = []
        for x in splits:
            x = tf.squeeze(x, axis=1)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.A_inter])
                x = temp + x
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.A_inter])
                temp = temp + x
            xcoll.append(temp)
        x2_inter = tf.concat(xcoll, axis=1)

        if self.track_feat:
            self.x2_inter = tf.identity(x2_inter)

        # Combinar y clasificar
        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        x3 = self.roi_droput_2(x2)
        xf = self.GlobAttpool(x3)

        if self.track_feat:
            self.GlobAttpool_feat = tf.identity(xf)

        xf = self.BN2(xf)
        out = self.Dense(xf)
        return out

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

