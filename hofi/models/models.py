import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM, Dense, Lambda, Conv2D
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, APPNPConv, GATConv # diffent convolution layers
from spektral.layers import GlobalAttentionPool, GlobalAttnSumPool # different pooling layers
from spektral.utils import normalized_adjacency

# from user-defined scripts
from utils import RoiPoolingConv, getROIS, getIntegralROIS, crop, squeezefunc, stackfunc

################## BASE Class for Parameter Initialization ###########################
class Params(Model):
    """ PARAMETERS INITIALIZATION OF ALL MODElS """
    def __init__(
        self,
        pool_size=None,
        ROIS_resolution=None,
        ROIS_grid_size=None,
        minSize=None,
        alpha=None,
        nb_classes=None,
        batch_size=None,
        input_sh = (224, 224, 3),
        cnodes = 32,
        gcn_outfeat_dim = 256,
        gat_outfeat_dim = 256,
        dropout_rate = 0.2,
        l2_reg = 2.5e-4,
        attn_heads = 1,
        appnp_activation = 'sigmoid',
        gat_activation = 'elu',
        concat_heads = True,
        backbone = None,
        freeze_backbone = None,
        gnn1_layr = True,
        gnn2_layr = True,
        track_feat = False,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.ROIS_resolution = ROIS_resolution
        self.ROIS_grid_size = ROIS_grid_size
        self.minSize = minSize
        self.alpha = alpha
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_sh = input_sh
        self.cnodes = cnodes
        self.gcn_outfeat_dim = gcn_outfeat_dim
        self.gat_outfeat_dim = gat_outfeat_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg  # L2 regularization rate
        self.attn_heads = attn_heads
        self.appnp_activation = appnp_activation
        self.gat_activation = gat_activation
        self.concat_heads = concat_heads
        self.gnn1_layr = gnn1_layr
        self.gnn2_layr = gnn2_layr
        self.track_feat = track_feat
            
        base_model_class = getattr(tf.keras.applications, backbone)  # Backbone call directly from tf.keras.applications
        self.base_model = base_model_class(
            weights="imagenet",
            input_tensor = layers.Input( shape = self.input_sh ), 
            include_top = False,
        )      
        
        # Freeze backbone for experimentation
        if freeze_backbone:
            for layer in self.base_model.layers:
                layer.trainable = False           

        if self.concat_heads:    # If concat_heads == True, then split channels over heads -->  node_dim == outfeat_dim 
            self.gat_outfeat_dim = self.gat_outfeat_dim // self.attn_heads
            

# ################################################################################ #
# ############################# Model Definations ################################ #
# ################################################################################ #

class BASE_CNN(Params):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Build required top layers
        self._construct_layers()
        
    def _construct_layers(self):
        # Add a custom classification head
        self.GAP_layer = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.nb_classes, activation="softmax")  #  nb_classes, softmax activation
        
    def call(self, inputs):
        base_out = self.base_model(inputs)
        x_gap = self.GAP_layer(base_out)
        x = self.dense(x_gap)
        
        # Track features for t-SNE computation
        if self.track_feat:
            self.base_out = tf.identity(base_out)
            self.GlobAttpool_feat = tf.identity(x_gap)
        
        return x

##################################### GATConv, SC_GNN_COMB ########################################
class GATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print('==========> CUSTOM GATconv Class <==========')

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        if self.add_self_loops:
            a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = tf.where(K.eval(a) == 0.0, -10e9, 0.0)
        mask = tf.cast(mask, dtype=attn_coef.dtype)

        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

        return output, attn_coef

class I2HOFI(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print(' ============= Model Name ::::: SC_(APPNP + GAT)_COMB with RESIDUAL (with Dense) =================')
        # print(self.l2_reg, self.attn_heads, self.dropout_rate)

        dims = self.base_model.output.shape.as_list()[1:]
        self.base_channels = dims[2]
        self.feat_dim = int(self.base_channels) * self.pool_size * self.pool_size
        # print('----> self.base_channels', self.base_channels)
        # print('----> self.pool_size', self.pool_size)


        """ Do the ROIs information and separate them out """
        self.rois_mat =  getROIS(
            resolution = self.ROIS_resolution,
            gridSize = self.ROIS_grid_size, 
            minSize = self.minSize
            )
        # rois_mat = getIntegralROIS()
        self.num_rois = self.rois_mat.shape[0]

        # Additional initialization logic
        self.cnodes = (int(self.base_channels) // self.gcn_outfeat_dim) * self.pool_size * self.pool_size

        # Construct the adjecency matrices
        self._construct_adjecency()

        # Build required layers
        self._construct_layers()

    # Rest of the class definition
    def _construct_adjecency(self):
        # super()._construct_adjecency()

        A1 = np.ones((self.cnodes, self.cnodes), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr1 = GCNConv.preprocess(A1).astype('f4')
        A_intra = Input(tensor=sp_matrix_to_sp_tensor(cfltr1), name = 'AdjacencyMatrix1') 

        A2 = np.ones((self.num_rois + 1, self.num_rois + 1), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr2 = GCNConv.preprocess(A2).astype('f4')
        A_inter = Input(tensor=sp_matrix_to_sp_tensor(cfltr2), name = 'AdjacencyMatrix2') 

        self.Adj = [A_intra, A_inter]
        # print('////////////// shape pf adj1, adj2', self.Adj[0].shape, self.Adj[1].shape)

    def _temp_nodes_tranform(self, roi):
        # print('shape of roi to transform ------------>', roi.shape)
        # Reshape the tensor
        reshaped_data = tf.reshape(roi, (-1, self.pool_size * self.pool_size, self.base_channels))

        # print('shape of reshaped_data ------------>', reshaped_data.shape, int(self.base_channels) // self.gcn_outfeat_dim, int(self.base_channels), self.gcn_outfeat_dim)

        # split in channels
        splits = tf.split(reshaped_data, num_or_size_splits = int(self.base_channels) // self.gcn_outfeat_dim , axis = 2)
        # print('no of splits ---------> {} | each with shape -----------> {}'.format(len(splits), splits[0].shape), self.cnodes)

        # combine into nodes
        joined = tf.concat(splits, 1)
        # print('shape of joined ------------>', joined.shape)

        return joined

    def _construct_layers(self):
        """ Override _construct_layers method with modified layer definitions """
        # super()._construct_layers()  # Call the parent class method to retain its functionality

        """ Different layer definations """
        self.upsampling_layer = layers.Lambda(lambda x: tf.image.resize(x, size = (self.ROIS_resolution, self.ROIS_resolution)), name = 'UpSample')
        self.roi_pooling = RoiPoolingConv(pool_size = self.pool_size, num_rois = self.num_rois, rois_mat = self.rois_mat)

        """ Dropout layers (after ROI pooling) """
        self.roi_droput_1 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_1')

        """  Time distributed layer applied to roi pooling """
        self.timedist_layer1 = layers.TimeDistributed(
            layers.Reshape((self.pool_size, self.pool_size, self.base_channels)), name='TD_Layer1'
        )

        self.timedist_layer2 = layers.TimeDistributed(
            layers.Lambda(self._temp_nodes_tranform), name='TD_Layer2'
        )

        ######### Temporal GCN layers
        self.tgcn_1 = APPNPConv(
            self.gcn_outfeat_dim, 
            alpha = self.alpha, 
            propagations = 1, 
            mlp_activation = self.appnp_activation, 
            use_bias = True, 
            name = 'GNN_1'
        )

        self.tgcn_2 = GATConv(
            self.gat_outfeat_dim,
            attn_heads = self.attn_heads,
            concat_heads = self.concat_heads,
            dropout_rate = self.dropout_rate,
            activation = self.gat_activation,
            kernel_regularizer = l2(self.l2_reg),
            attn_kernel_regularizer = l2(self.l2_reg),
            bias_regularizer = l2(self.l2_reg), 
            name = 'GNN_2'
            )

        """ Dropout layer (after combining all nodes from inter and intra) """       
        self.roi_droput_2 = tf.keras.layers.Dropout(self.dropout_rate, name='DOUT_2')

        """ Final layers """
        self.GlobAttpool = GlobalAttentionPool(self.gcn_outfeat_dim * 2, name = 'GlobalAttnPool')
        self.BN2 = layers.BatchNormalization(name = 'BN')
        self.Dense = layers.Dense(self.nb_classes, activation='softmax', name='Fully_Conn')


    def _extract_roi_nodes(self, x0, base_out):
        ########################################### """ ROI pooling and sequential self-attention """ 
        roi_pool = self.roi_pooling(x0)

        # print('roi_pool in _extract_roi_nodes ~~~~~~~~~~~~~~~~~>', roi_pool)
        # print('~~~~~~~~~~~~~~~~~~ From extract roi nodes function ~~~~~~~~~~~~~~', self.num_rois, self.feat_dim, self.pool_size)
        # print('~~~~~~~ self.roi_droput_1~~~~~~~', self.roi_droput_1)

        jcvs = []
        for j in range(self.num_rois):
            roi_crop = crop(1, j, j + 1)(roi_pool)
            #roi_crop.name = 'lambda_crop_'+str(j)
            #print(roi_crop)
            lname = 'roi_lambda_' + str(j)
            x = layers.Lambda(squeezefunc, name = lname)(roi_crop)
            
            # x = tf.reshape(x, (-1, self.feat_dim))
            x = layers.Reshape((self.feat_dim,))(x)
            
            jcvs.append(x)

        if self.pool_size != base_out.shape[1]: # Resize the original based on pool_size
            base_out = layers.Lambda(lambda x: tf.image.resize(x, size = (self.pool_size, self.pool_size)), name = 'Lambda_img_2')(base_out)

        # x = tf.reshape(base_out, (-1, self.feat_dim))
        x = layers.Reshape((self.feat_dim,))(base_out) # append the original ones
        
        jcvs.append(x)
        jcvs = layers.Lambda(stackfunc, name = 'lambda_stack')(jcvs)
        jcvs = self.roi_droput_1(jcvs)

        return jcvs
        
    def call(self, inputs):
        # Input to base model
        base_out = self.base_model(inputs)

        # Track base features for t-SNE computation
        if self.track_feat:  
            self.base_out = tf.identity(base_out)
        
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        # print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        # print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        # print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)
        # print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Intra
        splits = tf.split(x1, num_or_size_splits = self.num_rois + 1, axis = 1)
        xcoll = []
        for x in splits:
            # print('/////////////////////////////////////')
            x = tf.squeeze(x, axis=1)
            # print('Shape of each roi x --> ', x.shape)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj[0] ])
                x = temp + x # residual connection
            # print('Shape of each roi temp after tgcn_1 --> ',temp.shape)
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj[0] ])
                temp = temp + x
            # print('Shape of each roi temp after tgcn_2 (intra) --> ',temp.shape)
            xcoll.append(temp)

        x2_intra = tf.concat(xcoll, axis=1)
        # print('x2 shape after  self.tgcn_2 (intra) ---> ', x2_intra.shape )
        
        # Track intra-roi features for t-SNE computation
        if self.track_feat:
            self.x2_intra = tf.identity(x2_intra)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inter 
        # permute layer to swap nodes <--> graphs
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        # print('x1 shape after permutation ---> ', x1.shape, K.eval(tf.rank(x1)), self.cnodes )

        splits = tf.split(x1, num_or_size_splits = self.cnodes, axis = 1)
        xcoll = []
        for x in splits:
            # print('/////////////////////////////////////')
            x = tf.squeeze(x, axis=1)
            # print('Shape of each roi x --> ', x.shape)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj[1] ])
                x = temp + x # residual connection
            # print('Shape of each roi temp after tgcn_1 --> ',temp.shape)
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj[1] ])
                temp = temp + x
            # print('Shape of each roi temp after tgcn_2 (Inter) --> ',temp.shape)
            xcoll.append(temp)

        x2_inter = tf.concat(xcoll, axis=1)
        # print('x2 shape after  self.tgcn_2 (inter) ---> ', x2_inter.shape )
        
        # Track inter-roi features for t-SNE computation
        if self.track_feat:
            self.x2_inter = tf.identity(x2_inter)
        # ~~~~~~~~~~~~~~~~~~~~~~

        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        # print('x2 shape after  self.tgcn_2 (intra + inter) ---> ', x2.shape )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x3 = self.roi_droput_2(x2)
        # print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        # print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))
        
        # Track Gated Attention pooled features for t-SNE computation
        if self.track_feat:
            print('__________ track_feat 04 is active!')
            self.GlobAttpool_feat = tf.identity(xf)

        xf = self.BN2(xf)
    
        feat = self.Dense(xf)

        # print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )

        return feat
