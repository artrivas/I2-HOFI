import numpy as np
from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, APPNPConv, ARMAConv, GATConv, GINConv, GraphSageConv # diffent convolution layers
from spektral.layers import GlobalAttentionPool, SortPool, TopKPool, GlobalSumPool, GlobalAttnSumPool # different pooling layers
from spektral.layers import MinCutPool, DMoNPool

from spektral.utils import normalized_adjacency
# from keras_self_attention import SeqSelfAttention
# from keras_self_attention import SeqWeightedAttention as Attention

import tensorflow as tf
# from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM, Dense, Lambda, Conv2D
from tensorflow.keras import layers, Model
# from tensorflow.keras.models import Model #load_model
import os

# from RoiPoolingConvTF2 import RoiPoolingConv
from utils import RoiPoolingConv, getROIS, getIntegralROIS, crop, squeezefunc, stackfunc

import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

from .models import *


################## ALL Parameters in seperate class file ###########################

class SC_GNN_Params(Model):
    """ PARAMETERS OF ALL THE MODEL FILES ALLTOGETHER """
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
        
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKBONE CALL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ', backbone)       
        base_model_class = getattr(tf.keras.applications, backbone)
        self.base_model = base_model_class(
            weights="imagenet",
            input_tensor = layers.Input( shape = self.input_sh ), 
            include_top = False,
        )
        # print('============= self.base_model ===================',self.base_model.summary())       
        
        # Freeze backbone for experimentation
        if freeze_backbone:
            for layer in self.base_model.layers:
                layer.trainable = False
                
                
        # Check and rouning of the final dimension based on backbone model       
        def modify_final_dim(base_model, new_channel_dim):
            """Modify the final convolutional layer based on the backbone model and desired number of filters."""
            x = base_model.layers[-1].output
            x = Conv2D(
                new_channel_dim,
                (1, 1),
                strides=(1, 1),
                padding='same',
                activation='relu',
                name='new_conv_rounding_outdim'
            )(x)
            base_model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
            return base_model
        
        if backbone == 'NASNetMobile':
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$ Rounding outdim of NASNetMobile $$$$$$$$$$$$$$$$$$$$$$$')
            self.base_model = modify_final_dim(self.base_model, 1024)
        elif backbone == 'NASNetLarge':
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$ Rounding outdim of NASNetLarge $$$$$$$$$$$$$$$$$$$$$$$')
            self.base_model = modify_final_dim(self.base_model, 4096)
        elif backbone == 'MobileNetV3Large':
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$ Rounding outdim of MobileNetV3Small $$$$$$$$$$$$$$$$$$$$$$$')
            self.base_model = modify_final_dim(self.base_model, 1024)
            

        if self.concat_heads:
            self.gat_outfeat_dim = self.gat_outfeat_dim // self.attn_heads
            print('````````````` concat_heads : True `````````````` GATconv attention head : {} | self.gcn_outfeat_dim :{}'.format(self.attn_heads, self.gat_outfeat_dim))#

# ################################################################################ #
# ############################# Model Definations ################################ #
# ################################################################################ #

class BASE_CNN(SC_GNN_Params):
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
        
        track_bool = True
        if track_bool:
            self.base_out = tf.identity(base_out)
            self.GlobAttpool_feat = tf.identity(x_gap)
        
        return x

 ################## The BASECLASS defination ###########################

class SC_GNN(SC_GNN_Params):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Rest of the initialization logic
        self.message = " "

        dims = self.base_model.output.shape.as_list()[1:]
        self.base_channels = dims[2]
        self.feat_dim = int(self.base_channels) * self.pool_size * self.pool_size
        print('----> self.base_channels', self.base_channels)
        print('----> self.pool_size', self.pool_size)


        """ Do the ROIs information and separate them out """
        self.rois_mat =  getROIS(
            resolution = self.ROIS_resolution,
            gridSize = self.ROIS_grid_size, 
            minSize = self.minSize
            )
        # rois_mat = getIntegralROIS()
        self.num_rois = self.rois_mat.shape[0]

        # print('----------------------------------- self.rois_mat ------------------------------> ', self.rois_mat, np.shape(self.rois_mat))


        # Construct the adjecency matrices
        self._construct_adjecency()

        # Build required layers
        self._construct_layers()

    # Rest of the class definition
    def _construct_adjecency(self):
        """ Create, preprocess and combine Adjacency matrix """
        # N = int(full_img.shape[-1]) # total no of channels present  
        CA = np.ones((self.cnodes, self.cnodes), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr = GCNConv.preprocess(CA).astype('f4')
        CA_in = Input(tensor=sp_matrix_to_sp_tensor(cfltr), name = 'ChannelAdjacencyMatrix') 

        # N1 = self.num_rois + 1
        A = np.ones((self.num_rois + 1 , self.num_rois + 1), dtype = 'int')
        fltr = GCNConv.preprocess(A).astype('f4')
        A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr), name = 'AdjacencyMatrix')

        self.Adj = [CA_in, A_in]


    def _construct_layers(self):
        """ Different layer definations """
        self.upsampling_layer = layers.Lambda(lambda x: tf.image.resize(x, size = (self.ROIS_resolution, self.ROIS_resolution)), name = 'Lambda_img_1')
        self.roi_pooling = RoiPoolingConv(pool_size = self.pool_size, num_rois = self.num_rois, rois_mat = self.rois_mat)

        ######### Temporal GCN layers
        self.tgcn_1 = APPNPConv(self.gcn_outfeat_dim, alpha = self.alpha, propagations = 1, mlp_activation = self.appnp_activation, use_bias = True, name = 'chan_gnn_1')
        self.tgcn_2 = APPNPConv(self.gcn_outfeat_dim, alpha = self.alpha, propagations = 1, mlp_activation = self.appnp_activation, use_bias = True, name = 'chan_gnn_2')

        ######### Spatial (ROI) GCN layers
        self.sgcn_1 = APPNPConv(self.gcn_outfeat_dim, alpha = self.alpha, propagations = 1, mlp_activation = self.appnp_activation, use_bias = True, name = 'roi_gnn_1')
        self.sgcn_2 = APPNPConv(self.gcn_outfeat_dim, alpha = self.alpha, propagations = 1, mlp_activation = self.appnp_activation, use_bias = True, name = 'roi_gnn_2')


        """  Time distributed layer applied to roi pooling """
        self.timedist_layer1 = layers.TimeDistributed(layers.Reshape((self.pool_size, self.pool_size, self.base_channels)), name='td_layer1')
        self.timedist_layer_GAP = layers.TimeDistributed(layers.GlobalAveragePooling2D(name = 'GAP_time'))

        """ Dropout layers """
        self.channel_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.roi_droput_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.roi_droput_2 = tf.keras.layers.Dropout(self.dropout_rate)

        """ Final layers """
        self.GlobAttpool = GlobalAttentionPool(self.gcn_outfeat_dim * 2, name = 'GlobalAttnPool')
        self.BN2 = layers.BatchNormalization(name = 'BN2')
        self.Dense = layers.Dense(self.nb_classes, activation='softmax', name='FinalDense')
        
    # @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def _temp_nodes_tranform(self, x0):
        ############################################# """  Tranform apply channel GCN  """ (can seperate transform into )
        transposed_tensor = tf.transpose(x0, perm = [0, 3, 1, 2]) # Convert to channel first (None, 2048, 42, 42)
        splits = tf.split(transposed_tensor, num_or_size_splits = self.cnodes, axis = 1) # Split into cnodes components

        # Flatten each split
        flattened_splits = [tf.reshape(x, shape = (-1, (self.base_channels // self.cnodes) * self.ROIS_resolution * self.ROIS_resolution)  ) for x in splits]
        flattened_splits = [tf.expand_dims(x, 1) for x in flattened_splits]

        # Concatenate into single tensor
        joined = tf.concat(flattened_splits, 1)

        return joined

    
    # @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def _extract_roi_nodes(self, x0, base_out):
        ########################################### """ ROI pooling and sequential self-attention """ 
        roi_pool = self.roi_pooling(x0)

        # print('roi_pool in _extract_roi_nodes ~~~~~~~~~~~~~~~~~>', roi_pool)
        print('~~~~~~~~~~~~~~~~~~ From extract roi nodes function ~~~~~~~~~~~~~~', self.num_rois, self.feat_dim, self.pool_size)
        print('~~~~~~~ self.roi_droput_1~~~~~~~', self.roi_droput_1)

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
        """ Backbone model """
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])

        
        # """ infering other parameters """
        # self.feat_dim = int(x0.shape[-1]) * self.pool_size * self.pool_size

        """  << Channel-wise GCN >> : tranform into nodes and apply Channel-wise GCN  """ 
        c0 = self._temp_nodes_tranform(x0)
        print('============ c0 shape, self.Adj[0].shape', c0.shape, self.Adj[0].shape)
        c0 = self.tgcn_1([c0, self.Adj[0]])
        c0 = self.tgcn_2([c0, self.Adj[0]])    
        c1 = self.channel_dropout(c0) # Adding a dropout ---> newly added
        print('c1 shape ---> ', c1.shape)


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        x1 = self.timedist_layer1(rois)
        x1 = self.timedist_layer_GAP(x1)

        """ Applying spatial ROI based GCN model """
        x2 = self.sgcn_1([x1, self.Adj[1]])
        x2 = self.sgcn_2([x2, self.Adj[1]])
        x3 = self.roi_droput_2(x2)

        """ Now combine x0 and x3 through spectral clustering """
        all_nodes = layers.Concatenate(axis = 1)([c1, x3])
        print('all_nodes shape ---> ', all_nodes.shape)

        xf = self.GlobAttpool(all_nodes)
        print('xf shape after GAattP ---> ', xf.shape)

        xf = self.BN2(xf)

        feat = self.Dense(xf)
        print('xf shape after Dense (new) ---> ', feat.shape )
        
        return feat

##################################### SC_GNN_INTRA ########################################

class SC_GNN_INTRA(SC_GNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name :::::::: SC_GNN_INTRA (with Dense) [new param logic] =================')
        
        # Additional initialization logic
        self.cnodes = (int(self.base_channels) // self.gcn_outfeat_dim) * self.pool_size * self.pool_size

        # Construct the adjecency matrices
        self._construct_adjecency()

        # Build required layers
        self._construct_layers()

        print("/***************** Feature dimension ************************* /", self.gcn_outfeat_dim)

    # Rest of the class definition
    def _construct_layers(self):
        """ Override _construct_layers method with modified layer definitions """
        super()._construct_layers()  # Call the parent class method to retain its functionality

        self.timedist_layer2 = layers.TimeDistributed( 
            layers.Lambda(self._temp_nodes_tranform), name='td_layer2' #layers.Reshape((self.pool_size, self.pool_size, self.base_channels)) 
            )
        
        # # <<< Layers added for replacing last Dense >>>
        # self.GlobAttpool = GlobalAttentionPool(self.nb_classes, name = 'GlobalAttnPool')
        # self.softmax_act = layers.Activation('softmax')

        # removing redundant layer defination from parent class
        del self.sgcn_1
        del self.sgcn_2
        del self.timedist_layer_GAP

    def _construct_adjecency(self):
        super()._construct_adjecency()

        A = np.ones((self.cnodes, self.cnodes), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr = GCNConv.preprocess(A).astype('f4')
        A_in = Input(tensor=sp_matrix_to_sp_tensor(cfltr), name = 'AdjacencyMatrix') 

        self.Adj = A_in

    def _temp_nodes_tranform(self, roi):
        print('shape of roi to transform ------------>', roi.shape)
        # Reshape the tensor
        reshaped_data = tf.reshape(roi, (-1, self.pool_size * self.pool_size, self.base_channels))

        print('shape of reshaped_data ------------>', reshaped_data.shape, int(self.base_channels) // self.gcn_outfeat_dim, int(self.base_channels), self.gcn_outfeat_dim)

        # split in channels
        splits = tf.split(reshaped_data, num_or_size_splits = int(self.base_channels) // self.gcn_outfeat_dim , axis = 2)
        print('no of splits ---------> {} | each with shape -----------> {}'.format(len(splits), splits[0].shape), self.cnodes)

        # combine into nodes
        joined = tf.concat(splits, 1)
        print('shape of joined ------------>', joined.shape)

        return joined


    def call(self, inputs):
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1) # Here --> split in channels
        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )

        splits = tf.split(x1, num_or_size_splits = self.num_rois + 1, axis = 1)
        xcoll = []
        for x in splits:
            temp = self.tgcn_1([tf.squeeze(x, axis=1), self.Adj])
            temp = self.tgcn_2([temp, self.Adj])
            xcoll.append(temp)

        x2 = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_2---> ', x2.shape )


        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))

        xf = self.BN2(xf)
        
        feat = self.Dense(xf)
        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )


        return feat
    
########################### GNN + RESIDUAL (Intra) ##################################    
class SC_GNNRES_INTRA(SC_GNN_INTRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        print(' ============= Model Name :::::::: SC_GNN_INTRA with RESIDUAL Connection (with Dense) =================', self.gnn1_layr, self.gnn2_layr) 

    def call(self, inputs):
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)
        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )


        splits = tf.split(x1, num_or_size_splits = self.num_rois + 1, axis = 1)
        xcoll = []
        for x in splits:
            # print('/////////////////////////////////////')
            x = tf.squeeze(x, axis=1)
            # print('Shape of each roi x --> ', x.shape)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj])
                x = temp + x # residual connection
            # print('Shape of each roi temp after tgcn_1 --> ',temp.shape)
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj])
                temp = temp + x
            # print('Shape of each roi temp after tgcn_2 --> ',temp.shape)
            xcoll.append(temp)

        x2 = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_2---> ', x2.shape )


        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))

        xf = self.BN2(xf)
        
        # feat = self.softmax_act(xf)

        feat = self.Dense(xf)
        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )


        return feat
    
############################## GNN + INTER ##################################
######################### GNN + RESIDUAL + INTER ##################################

class SC_GNN_INTER(SC_GNN_INTRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name :::::::: SC_GNN_INTER (with Dense) =================')

        # Construct the adjecency matrices
        self._construct_adjecency()

        # Re-Build required layers
        self._construct_layers()
    

    def _construct_adjecency(self):
        super()._construct_adjecency()

        A = np.ones((self.num_rois + 1, self.num_rois + 1), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr = GCNConv.preprocess(A).astype('f4')
        A_in = Input(tensor=sp_matrix_to_sp_tensor(cfltr), name = 'AdjacencyMatrix') 

        self.Adj = A_in


    def call(self, inputs):
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)
        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )

        # permute layer to swap nodes <--> graphs
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        print('x1 shape after permutation ---> ', x1.shape, K.eval(tf.rank(x1)), self.cnodes )

        splits = tf.split(x1, num_or_size_splits = self.cnodes, axis = 1)
        xcoll = []
        for x in splits:
            temp = self.tgcn_1([tf.squeeze(x, axis=1), self.Adj])
            temp = self.tgcn_2([temp, self.Adj])
            xcoll.append(temp)

        x2 = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_1---> ', x2.shape )

        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))

        xf = self.BN2(xf)
        
        feat = self.Dense(xf)

        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )

        return feat


class SC_GNNRES_INTER(SC_GNN_INTER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        print(' ============= Model Name :::::::: SC_GNN_INTER with RESIDUAL Connection (with Dense) =================')


    def call(self, inputs):
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)

        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )

        # permute layer to swap nodes <--> graphs
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        print('x1 shape after permutation ---> ', x1.shape, K.eval(tf.rank(x1)), self.cnodes )


        splits = tf.split(x1, num_or_size_splits = self.cnodes, axis = 1)
        xcoll = []
        for x in splits:
            # print('/////////////////////////////////////')
            x = tf.squeeze(x, axis=1)
            # print('Shape of each roi x --> ', x.shape)
            if self.gnn1_layr:
                temp = self.tgcn_1([x, self.Adj])
                x = temp + x # residual connection
            # print('Shape of each roi temp after tgcn_1 --> ',temp.shape)
            if self.gnn2_layr:
                temp = self.tgcn_2([x, self.Adj])
                temp = temp + x
            # print('Shape of each roi temp after tgcn_2 --> ',temp.shape)
            xcoll.append(temp)

        x2 = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_1---> ', x2.shape )

        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))

        xf = self.BN2(xf)
        
        # feat = self.softmax_act(xf)
        feat = self.Dense(xf)

        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )

        return feat 


##########################################################################################
##################################### SC_GNN_COMB ########################################
class SC_GNN_COMB(SC_GNN_INTRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name :::::::: SC_GNN_INTER - INTRA combined (with Dense), APPNP layers shared [new param logic] =================')
        # Additional initialization logic
        # Construct the adjecency matrices
        self._construct_adjecency()


    # Rest of the class definition
    def _construct_adjecency(self):
        super()._construct_adjecency()

        A1 = np.ones((self.cnodes, self.cnodes), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr1 = GCNConv.preprocess(A1).astype('f4')
        A_intra = Input(tensor=sp_matrix_to_sp_tensor(cfltr1), name = 'AdjacencyMatrix1') 

        A2 = np.ones((self.num_rois + 1, self.num_rois + 1), dtype = 'int') # CA = np.ones((N,N), dtype='int')
        cfltr2 = GCNConv.preprocess(A2).astype('f4')
        A_inter = Input(tensor=sp_matrix_to_sp_tensor(cfltr2), name = 'AdjacencyMatrix2') 

        self.Adj = [A_intra, A_inter]
        print('////////////// shape pf adj1, adj2', self.Adj[0].shape, self.Adj[1].shape)


    def call(self, inputs):
        base_out = self.base_model(inputs)
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)

        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Intra

        splits = tf.split(x1, num_or_size_splits = self.num_rois + 1, axis = 1)
        xcoll = []
        for x in splits:
            temp = self.tgcn_1([tf.squeeze(x, axis=1), self.Adj[0]])
            temp = self.tgcn_2([temp, self.Adj[0] ])
            xcoll.append(temp)

        x2_intra = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_2 (intra) ---> ', x2_intra.shape )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inter 
        # permute layer to swap nodes <--> graphs
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        print('x1 shape after permutation ---> ', x1.shape, K.eval(tf.rank(x1)), self.cnodes )

        splits = tf.split(x1, num_or_size_splits = self.cnodes, axis = 1)
        xcoll = []
        for x in splits:
            temp = self.tgcn_1([tf.squeeze(x, axis=1), self.Adj[1] ])
            temp = self.tgcn_2([temp, self.Adj[1] ])
            xcoll.append(temp)

        x2_inter = tf.concat(xcoll, axis=1)
        print('x2 shape after  self.tgcn_2 (inter) ---> ', x2_inter.shape )

        # ~~~~~~~~~~~~~~~~~~~~~~
        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        print('x2 shape after  self.tgcn_2 (intra + inter) ---> ', x2.shape )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))

        xf = self.BN2(xf)
        
        feat = self.Dense(xf)

        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )

        return feat



##################################### SC_GNN_COMB ########################################
class SC_GNNRES_COMB(SC_GNN_COMB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' =========== Model Name :::::: SC_GNN_COMB with RESIDUAL Connection (with Dense) =================')
        # Additional initialization logic

    # Rest of the class definition

    def call(self, inputs):
        track_bool = True  # boolean to track features or not
        base_out = self.base_model(inputs)
        if track_bool:
            self.base_out = tf.identity(base_out)
        
        x0 = self.upsampling_layer(base_out) # x0 = full_image
        print('x0 shape, x0.shape[-1]', x0.shape, x0.shape[-1])


        """ << roi Spatial - GCN >> :Applying Reshape and 2D GlobalAveragepooling to each ROI """ 
        rois = self._extract_roi_nodes(x0, base_out)
        print('rois shape  ---> ', rois.shape )

        x1 = self.timedist_layer1(rois)
        print('x1 shape after  timedist_layer1 ---> ', x1.shape )

        x1 = self.timedist_layer2(x1)

        print('x1 shape after timedist_layer2 ---> ', x1.shape, K.eval(tf.rank(x1)) )

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
        print('x2 shape after  self.tgcn_2 (intra) ---> ', x2_intra.shape )
        if track_bool:
            self.x2_intra = tf.identity(x2_intra)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inter 
        # permute layer to swap nodes <--> graphs
        x1 = tf.transpose(x1, perm=[0, 2, 1, 3])
        print('x1 shape after permutation ---> ', x1.shape, K.eval(tf.rank(x1)), self.cnodes )

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
        print('x2 shape after  self.tgcn_2 (inter) ---> ', x2_inter.shape )
        if track_bool:
            self.x2_inter = tf.identity(x2_inter)
        # ~~~~~~~~~~~~~~~~~~~~~~

        x2 = tf.concat([x2_intra, x2_inter], axis=1)
        print('x2 shape after  self.tgcn_2 (intra + inter) ---> ', x2.shape )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x3 = self.roi_droput_2(x2)
        print('x3 shape after  roi_droput_2 ---> ', x3.shape)

        xf = self.GlobAttpool(x3)
        print('xf shape after GAattP ---> ', xf.shape, K.eval(tf.rank(xf)))
        if track_bool:
            self.GlobAttpool_feat = tf.identity(xf)

        xf = self.BN2(xf)
    
        feat = self.Dense(xf)

        print('feat shape without Dense ---> ', feat.shape, K.eval(tf.rank(feat)) )

        return feat


##################################### GATConv, SC_GNN_COMB ########################################
class GATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('==========> CUSTOM GATconv Class <==========')

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
    
    
class SC_APPNPGATRES_INTRA(SC_GNNRES_INTRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name ::::: SC_(APPNP + GAT)_INTRA with RESIDUAL (with Dense) =================')
        print(self.l2_reg, self.attn_heads, self.dropout_rate)

        # Build required layers
        self._construct_layers()

    # Rest of the class definition
    def _construct_layers(self):
        """ Override _construct_layers method with modified layer definitions """
        super()._construct_layers()  # Call the parent class method to retain its functionality

        self.tgcn_2 = GATConv(
            self.gat_outfeat_dim,
            attn_heads = self.attn_heads,
            concat_heads = self.concat_heads,
            dropout_rate = self.dropout_rate,
            activation = self.gat_activation,
            kernel_regularizer = l2(self.l2_reg),
            attn_kernel_regularizer = l2(self.l2_reg),
            bias_regularizer = l2(self.l2_reg), 
            name = 'chan_gnn_2'
            )
        
        
class SC_APPNPGATRES_INTER(SC_GNNRES_INTER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name ::::: SC_(APPNP + GAT)_INTER with RESIDUAL (with Dense) =================')
        print(self.l2_reg, self.attn_heads, self.dropout_rate)

        # Build required layers
        self._construct_layers()

    # Rest of the class definition
    def _construct_layers(self):
        """ Override _construct_layers method with modified layer definitions """
        super()._construct_layers()  # Call the parent class method to retain its functionality

        self.tgcn_2 = GATConv(
            self.gat_outfeat_dim,
            attn_heads = self.attn_heads,
            concat_heads = self.concat_heads,
            dropout_rate = self.dropout_rate,
            activation = self.gat_activation,
            kernel_regularizer = l2(self.l2_reg),
            attn_kernel_regularizer = l2(self.l2_reg),
            bias_regularizer = l2(self.l2_reg), 
            name = 'chan_gnn_2'
            )


class SC_APPNPGATRES_COMB(SC_GNNRES_COMB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(' ============= Model Name ::::: SC_(APPNP + GAT)_COMB with RESIDUAL (with Dense) =================')
        print(self.l2_reg, self.attn_heads, self.dropout_rate)

        # Build required layers
        self._construct_layers()

    # Rest of the class definition
    def _construct_layers(self):
        """ Override _construct_layers method with modified layer definitions """
        super()._construct_layers()  # Call the parent class method to retain its functionality

        self.tgcn_2 = GATConv(
            self.gat_outfeat_dim,
            attn_heads = self.attn_heads,
            concat_heads = self.concat_heads,
            dropout_rate = self.dropout_rate,
            activation = self.gat_activation,
            kernel_regularizer = l2(self.l2_reg),
            attn_kernel_regularizer = l2(self.l2_reg),
            bias_regularizer = l2(self.l2_reg), 
            name = 'chan_gnn_2'
            )