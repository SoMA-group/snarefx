import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, MaxPooling2D, Flatten, \
    Activation, Permute, Concatenate, Reshape, GlobalAveragePooling2D, GlobalMaxPool2D, \
    MaxPool2D, concatenate, AveragePooling2D, GlobalAveragePooling1D, GlobalMaxPool1D,  \
    Conv2D, BatchNormalization, Layer
from tensorflow.keras import Model, layers
from tensorflow.keras.regularizers import l2

from deepafx.pool_loss_map.autopool import AutoPool1D

from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras import backend as K


class SqueezeLayer(Layer):
    '''
    Keras squeeze layer
    '''
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        # shape = np.array(input_shape)
        # shape = shape[shape != 1]
        # return tuple(shape)
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        return K.squeeze(x, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def backbone(x, plearn):

    # define convolutional stack: magic happens here
    # ...
    # ...
    # ...
    # define convolutional stack: magic happens here

    # final summarization pooling
    # IMPORTANT: what follows expects a fmap of shape (None, time, freq, channels) coming from the previous conv stack
    # IMPORTANT: what follows expects a fmap of shape (None, time, freq, channels) coming from the previous conv stack
    

    # ========BASIC POOLINGS=======================
    if plearn.get('global_pooling') == 'gmp':
        x = GlobalMaxPool2D(name='gmp')(x)
    elif plearn.get('global_pooling') == 'gap':
        x = GlobalAveragePooling2D(name='gap')(x)
    elif plearn.get('global_pooling') == 'gapgmp':
        gap = GlobalAveragePooling2D(name='gap')(x)
        gmp = GlobalMaxPool2D(name='gmp')(x)
        # concat in the time dim so that it is already flattened
        x = tf.concat([gap, gmp], 1)
    elif plearn.get('global_pooling') == 'gapgmpstd':
        gap = GlobalAveragePooling2D(name='gap')(x)
        gmp = GlobalMaxPool2D(name='gmp')(x)
        # yields nan sometimes, probably due to sqrt?
        # (None, time, freq), hence temporal pooling is axis=1 (rows)
        # (None, time, freq), hence global pooling is axis=[1, 2] (rows)
        # spec_std = tf.math.reduce_std(spec_rnn, axis=1, name='std')
        _, v = tf.nn.moments(x, axes=[1, 2], name='variance')
        std = tf.math.sqrt(v + tf.constant(1e-12), name='std')
        # concat in the time dim so that it is already flattened
        x = tf.concat([gap, gmp, std], 1)

    # ========TIME-FREQ POOLINGS=======================
    elif plearn.get('global_pooling') == 'meanmaxtime':
        # max temporal pooling followed by average freq pooling
        # (None, time, freq, channels)
        # vip default proposed oin c19_2
        freq_map = tf.reduce_max(x, axis=1)
        x = GlobalAveragePooling1D()(freq_map)
    elif plearn.get('global_pooling') == 'maxmeanfreq':
        # BEST TIME-FREQ POOLING==========================================
        # mean freq pooling followed by max temporal pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_mean(x, axis=2)
        x = GlobalMaxPool1D()(time_map)
    elif plearn.get('global_pooling') == 'meanmaxfreq':
        # max freq pooling followed by mean temporal pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_max(x, axis=2)
        x = GlobalAveragePooling1D()(time_map)
    elif plearn.get('global_pooling') == 'maxmeantime':
        # mean temporal pooling followed by max freq pooling
        # (None, time, freq, channels)
        freq_map = tf.reduce_mean(x, axis=1)
        x = GlobalMaxPool1D()(freq_map)

    # ========AUTOPOOL POOLINGS=======================
    # using AutoPool to learn a pooling operator for time, freq, or both
    # AutoPool is an adaptive (trainable) pooling operator which smoothly interpolates between common pooling operators,
    # such as min-, max-, or average-pooling, automatically adapting to the characteristics of the data.
    # AutoPool can be readily applied to any differentiable model for time-series label prediction.

    elif plearn.get('global_pooling') == 'automeanfreq':
        # BEST AUTOPOOL POOLING=======================
        # (None, time, freq, channels)
        time_map = tf.reduce_mean(x, axis=2)
        # we have a time series, (None, time, channels), that we want to pool
        # they had (None, time, n_class) to (None, n_class)
        x = AutoPool1D(axis=1)(time_map)
        # (None, channels)

    elif plearn.get('global_pooling') == 'meanautotime':
        # (None, time, freq, channels)
        freq_map = AutoPool1D(axis=1)(x)
        # we have a freq series, (None, freq, channels), that we want to pool
        x = GlobalAveragePooling1D()(freq_map)
        # (None, channels)

    elif plearn.get('global_pooling') == 'maxautofreq':
        # (None, time, freq, channels)
        time_map = AutoPool1D(axis=2)(x)
        # we have a time series, (None, time, channels), that we want to pool
        x = GlobalMaxPool1D()(time_map)
        # (None, channels)

    elif plearn.get('global_pooling') == 'automaxtime':
        # (None, time, freq, channels)
        freq_map = tf.reduce_max(x, axis=1)
        # we have a freq series, (None, freq, channels), that we want to pool
        x = AutoPool1D(axis=1)(freq_map)
        # (None, channels)

    elif plearn.get('global_pooling') == 'autoautofreq':
        # (None, time, freq, channels)
        time_map = AutoPool1D(axis=2)(x)
        # we have a time series, (None, time, channels), that we want to pool
        x = AutoPool1D(axis=1)(time_map)
        # (None, channels)

    elif plearn.get('global_pooling') == 'autoautotime':
        # (None, time, freq, channels)
        freq_map = AutoPool1D(axis=1)(x)
        # we have a freq series, (None, freq, channels), that we want to pool
        x = AutoPool1D(axis=1)(freq_map)
        # (None, channels)

    # ========MULTI-CHANNEL ATTENTION POOLINGS=======================
    elif plearn.get('global_pooling') == 'mcameanfreq':
        # BEST MULTI-CHANNEL ATTENTION POOLING=======================

        # mean freq pooling followed by multi-channel attention temporal pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_mean(x, axis=2)

        # (None, time, channels)
        # add multi-channel attention as in
        # https://arxiv.org/pdf/1805.03908.pdf and  https://arxiv.org/pdf/1910.12551v1.pdf

        # NOTE: what follows is for a conv layer with 256 channels. To be adjusted accordingly
        # x = torch.sum(x[:, :256] * torch.nn.functional.softmax(x[:, 256:], dim=3), dim=3, keepdim=True)
        # half of the filters (128) are input to a time-wise softmax activation,
        # which acts as an attentionmechanism for the other half of the filters
        # time-wise: axis=1
        # TODO: i've changed this to except variable number of filters
        x = tf.reduce_sum(time_map[:,:,:(x.shape[-1]//2)] * tf.nn.softmax(time_map[:,:,(x.shape[-1]//2):], axis=1, name='softmax_att'), axis=1, name='mca')
        # 128 time-only feature maps (13x128) * 128 sets of 13 attention (softmax) weights (13x128)
        # softmax returns the same type and shape as time_map, but squashed such as sums unity: attention weights
        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

        # (None, channels/2)

    elif plearn.get('global_pooling') == 'meanmcatime':
        # multi-channel attention temporal pooling, followed by average freq pooling
        # (None, time, freq, channels)
        freq_map = tf.reduce_sum(x[:,:,:,:(x.shape[-1]//2)] * tf.nn.softmax(x[:,:,:,(x.shape[-1]//2):], axis=1, name='softmax_att'), axis=1, name='mca')
        # (None, freq, channels/2)
        x = GlobalAveragePooling1D()(freq_map)
        # (None, channels/2)

    elif plearn.get('global_pooling') == 'maxmcafreq':
        # multi-channel attention freq pooling, followed by max temp pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_sum(x[:,:,:,:(x.shape[-1]//2)] * tf.nn.softmax(x[:,:,:,(x.shape[-1]//2):], axis=2, name='softmax_att'), axis=2, name='mca')
        # (None, time, channels/2)
        x = GlobalMaxPool1D()(time_map)
        # (None, channels/2)

    elif plearn.get('global_pooling') == 'mcamaxtime':
        #  max temp pooling, followed by multi-channel attention freq pooling
        # (None, time, freq, channels)
        freq_map = tf.reduce_max(x, axis=1)
        # (None, freq, channels)
        x = tf.reduce_sum(freq_map[:,:,:(x.shape[-1]//2)] * tf.nn.softmax(freq_map[:,:,(x.shape[-1]//2):], axis=1, name='softmax_att'), axis=1, name='mca')
        # (None, channels/2)
        
        
    elif plearn.get('global_pooling') == 'time_distributed_dense':
        """time distrubted dense + autopooling...
        only multi_layer works, rest don't
        """
        
        #TODO: classes or frames first ??  n_dense=5 or x.shape[1]...check this, read paper conv2d predictor
        
        
        
        # ADD THIS BACK IN FOR ALL MODELS BAR AUTOTAGGING
       ## x = MaxPooling2D((2, 2), padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        conv_squeeze = Conv2D(128, 
                              (1,x.shape[2]),
                              padding='valid',
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)

        squeeze = SqueezeLayer(axis=-2)(conv_squeeze)
        
        # #only use this for autotagging model
        # squeeze = SqueezeLayer(axis=-2)(x)
        

        dense_layer = tf.keras.layers.Dense(128,
                                            activation='sigmoid',
                                            kernel_initializer='he_normal',
                                            bias_regularizer=tf.keras.regularizers.l2(1e-5))

        td_layer = tf.keras.layers.TimeDistributed(dense_layer, name='td_layer')(squeeze)
        x = AutoPool1D(axis=1)(td_layer)

    # ========MAX+FLATTERN+DENSE=======================
    elif plearn.get('global_pooling') == 'flat_dense':
        #if we use this then need to consider if a models previous layer had maxpooling2d e.g. autotagging_new
        x = MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    return x


def get_model(pextract=None, plearn=None):

    # (None, time, freq, channels=1)
    input_shape = (pextract.get('patch_len'), pextract.get('n_mels'), 1)
    n_class = plearn.get('n_classes')
    x_in = Input(shape=input_shape)

    x = backbone(x_in, plearn)

    logits = Dense(n_class, name='logits', use_bias=True)(x)

    _model = Model(inputs=x_in, outputs=logits)

    return _model

