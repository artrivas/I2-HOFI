import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


#get regions of interest of an image (return all possible bounding boxes when splitting the image into a grid)
def getROIS(resolution=33,gridSize=3, minSize=1):
	
	coordsList = []
	step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
	for column1 in range(0, gridSize + 1):
		for column2 in range(0, gridSize + 1):
			for row1 in range(0, gridSize + 1):
				for row2 in range(0, gridSize + 1):
					
					#get coordinates using grid layout
					x0 = int(column1 * step)
					x1 = int(column2 * step)
					y0 = int(row1 * step)
					y1 = int(row2 * step)
					
					if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
						
						if not (x0==y0==0 and x1==y1==resolution): #ignore full image
							
							#calculate height and width of bounding box
							w = x1 - x0
							h = y1 - y0
							
							coordsList.append([x0, y0, w, h]) #add bounding box to list

	coordsArray = np.array(coordsList)	 #format coordinates as numpy array						

	return coordsArray
	
def getIntegralROIS(resolution=42,step=8, winSize=14):
    coordsList = []
    #step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
    for column1 in range(0, resolution, step):
        for column2 in range(0, resolution, step):
            for row1 in range(column1+winSize, resolution+winSize, winSize):
                for row2 in range(column2+winSize, resolution+winSize, winSize):
                    #get coordinates using grid layout
                    if row1 > resolution or row2 > resolution:
                        continue
                    x0 = int(column1)
                    y0 = int(column2)
                    x1 = int(row1)	
                    y1 = int(row2)	
                    
                    #if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
                    #    if not (x0==y0==0 and x1==y1==resolution): #ignore full image
                            #calculate height and width of bounding box
                    if not (x0==y0==0 and x1==y1==resolution):  #ignore full image
                        w = x1 - x0
                        h = y1 - y0
                        coordsList.append([x0, y0, w, h]) #add bounding box to list
    #coordsList.append([0, 0, resolution, resolution])#whole image
    coordsArray = np.array(coordsList)	 #format coordinates as numpy array
    return coordsArray	

def crop(dimension, start, end): #https://github.com/keras-team/keras/issues/890
    #Use this layer for a model that has individual roi bounding box
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)

def squeezefunc(x):
    return K.squeeze(x, axis=1)

'''This is to convert stacked tensor to sequence for LSTM'''
def stackfunc(x):
    return K.stack(x, axis=1) 

def get_flops(model=None, in_tensor=None):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            # Example model
            # model = tf.keras.applications.mobilenet.MobileNet(
            #         alpha=1, weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))

            model = model(in_tensor)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops


########################################################
if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, rois_mat, **kwargs):

        self.dim_ordering = "tf"
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois=rois_mat

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        
        #assert(len(x) == 2)

        #img = x[0]
        #rois = x[1]
        img = x

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            '''
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            '''
            x = self.rois[roi_idx, 0]
            y = self.rois[roi_idx, 1]
            w = self.rois[roi_idx, 2]
            h = self.rois[roi_idx, 3]
            
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        x2 = x1 + K.maximum(1,x2-x1)
                        y2 = y1 + K.maximum(1,y2-y1)
                        
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32') # K.maximum(K.cast(w, 'int32'), tf.constant(1))
                h = K.cast(h, 'int32') #K.maximum(K.cast(h, 'int32'), tf.constant(1))
                rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size)) #tf.cond(tf.logical_or(tf.equal(h, tf.constant(1)), tf.equal(w, tf.constant(1))), lambda: tf.constant(0.0, shape=(1,1,1)), lambda: tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size)))                #if w or h is == 0 then rs should be 0
                outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)) #first value must be -1 for batches bigger than 1!

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois,
                  'rois_mat': self.rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))