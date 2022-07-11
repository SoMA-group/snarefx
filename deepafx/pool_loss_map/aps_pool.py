#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:45:12 2021

@author: Eduardo Fonseca
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
# ##############settings:

# size=2
# stride=2
# mode = 'ApsPool'

# kernel_size = 1     # no blur RECOMMENDED
# # kernel_size = 3     # combine APS with BlurPool 3x3 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn')
# # kernel_size = 4     # combine APS with BlurPool 3x3 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn')
# # kernel_size = 5     # combine APS with BlurPool 5x5 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn') try as well, joints LPF and ApsPool together  RECOMMENDED

# apspool_criterion = 'l1' #  RECOMMENDED
# # apspool_criterion = 'l_infty'
# # apspool_criterion = 'l2'


# ##############call (equivalent to a conventional max-pool call):

# x = ApsPool(pool_size=size, pool_stride=stride, kernel_size=kernel_size, pool_mode=mode,
#             apspool_criterion=apspool_criterion)(x)



##############code:

ki = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)



class ApsPool(Layer):
    def __init__(self, pool_size: int = 2, pool_stride: int = 2, kernel_size: int = 1, pool_mode: str = 'max',
                 apspool_criterion='l2', **kwargs):

        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.blur_kernel = None          # later
        self.kernel_size = kernel_size   # for BP
        self.pool_mode = pool_mode
        self.apspool_criterion = apspool_criterion
        # self.return_poly_indices = return_poly_indices
        # Returning this index is useful in the bottleneck and basic block layers where we want the residual branch to
        # sample the same polyphase component index as the main branch.
        # APS at maxpool does not need to ensure this since there is no such residual branch there and hence it usesreturn_poly_indices = False.
        # we remove it from code for simplicity
        super(ApsPool, self).__init__(**kwargs)


    def build(self, input_shape):

        if self.kernel_size > 1:
            # construct blurring filter
            a = construct_1d_array(self.kernel_size)
            filt_np = a[:, None] * a[None, :]
            filt_np /= np.sum(filt_np)
            # this is for channel_last
            filt_np = np.repeat(filt_np[:, :, None], input_shape[-1], -1)

            # if K.image_data_format() == 'channels_first':
            #     filt_np = np.repeat(filt_np[None, :, :], input_shape[0], 0)
            # else:
            #     filt_np = np.repeat(filt_np[:, :, None], input_shape[-1], -1)
            filt_np = np.expand_dims(filt_np, -1)
            # (3, 3, 128, 1)
            blur_init = tf.keras.initializers.Constant(filt_np)

            if 'learn' in self.pool_mode:
                # vip ready softmax without constraint (done in call())
                # softmax constraints all weights to be positive and add up to unity.
                self.blur_kernel = self.add_weight(name='blur_kernel_learn',
                                                   shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                                   initializer=ki,
                                                   trainable=True)

            else:
                # blur_kernel_full_shape is strange: time, freq, channel, 1. This is required by depthwise_conv2d, which needs
                # a filter tensor of shape [filter_height, filter_width, in_channels, channel_multiplier]
                self.blur_kernel = self.add_weight(name='blur_kernel',
                                                   shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                                   initializer=blur_init,
                                                   trainable=False)

        # self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        # define here ReflectionPad2d or circular pad, later on over the incoming fmap
        self.permute_indices = None
        super(ApsPool, self).build(input_shape)  # Be sure to call this at the end


    def call(self, input_to_pool):

        if self.kernel_size > 1 and 'learn' in self.pool_mode:
            # only if we do learnable blurring
            # apply softmax
            low_pass_filter = tf.reshape(self.blur_kernel, [-1]) # flatten
            low_pass_filter = tf.nn.softmax(low_pass_filter, name='softmax_constr')    # ok
            low_pass_filter = tf.reshape(low_pass_filter, [self.kernel_size, self.kernel_size, -1, 1]) # back to squared shape OK
            # --
        else:
            low_pass_filter = self.blur_kernel

        # print(input_to_pool.shape)
        # 1) Apply dense max pooling evaluation (if original pooling is max-pooling)
        if 'avg' not in self.pool_mode and 'Avg' not in self.pool_mode:
            # dense max pooling evaluation, following original pooling size self.pool_size, and unit stride
            inp = tf.nn.pool(input_to_pool,
                           window_shape=(self.pool_size, self.pool_size),
                           strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')
        # Note: if the original pooling is avg pooling, there is nothing to be done in advance as
        # blurred downsampling with a box filter is already a form of avg pooling
        # we could try to do max_mean_freq here for summarization with antialiasing
        # in tf.nn.pool, instead of max, do it differently for each dim.
        # print(inp.shape)


        # 2) Apply APS, after the dense max-pool eval
        # this is the case when polyphase indices are not pre-defined
        polyphase_indices = None

        if self.kernel_size == 1:
            # only aps - no blurring
            return aps_downsample_v2(aps_pad(inp),                                      # pad to have even fmap
                                     self.pool_stride,                                       # stride
                                     polyphase_indices,                                 # polyphase_indices = None
                                     permute_indices=self.permute_indices,              # None
                                     apspool_criterion=self.apspool_criterion)          # l2

            # return aps_downsample_v2(aps_pad(inp), self.stride, polyphase_indices,
            #                          permute_indices=self.permute_indices,
            #                          apspool_criterion=self.apspool_criterion)

        else:
            # we do blurring followed by aps

            # 2) Apply blur_kernel but NOT subsample using depthwise_conv2d

            # strides in depthwise_conv performs the subsampling, which is usually 2x2
            # - we first apply low pass (antialiasing) filtering in both time and freq
            # - then we do NOT DO subsampling in both dimensions, as usual, specified with strides in depthwise_conv2d
            # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertical strides,
            strides = [1, 1, 1, 1]  # only blurring
            # subsampling done via aps
            blurred_inp = tf.nn.depthwise_conv2d(
                                                inp,
                                                low_pass_filter,   # self.blur_kernel,
                                                strides=strides,
                                                padding='SAME',
                                                data_format='NHWC',
                                                name='antialiasing_subsampling'
            )
            # Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a filter tensor of shape
            # [filter_height, filter_width, in_channels, channel_multiplier] containing in_channels convolutional filters of
            # depth 1, depthwise_conv2d applies a different filter to each input channel (expanding from 1 channel to
            # channel_multiplier channels for each), then concatenates the results together.
            # The output has in_channels * channel_multiplier channels.
            #
            return aps_downsample_v2(aps_pad(blurred_inp),                              # pad to have even fmap
                                     self.pool_stride,                                  # stride
                                     polyphase_indices,                                 # polyphase_indices = None
                                     permute_indices=self.permute_indices,              # None
                                     apspool_criterion=self.apspool_criterion)          # l2





def aps_downsample_v2(x, stride, polyphase_indices=None, permute_indices=None, apspool_criterion='l2'):
    """
        x is just a normal fmap. nothing special
    :param x:
    :param stride:
    :param polyphase_indices:
    :param permute_indices:
    :param apspool_criterion:
    :return:
    """
    # print('begin aps_downsample_v2', x.shape)

    # ctrl
    if stride == 1:
        return x

    elif stride > 2:
        raise Exception('Stride>2 currently not supported in this implementation')

    else:
        # only stride=2 is allowed

        # B, C, N, _ = x.shape
        # channel first in PyT, assumes squared images of size N. We need to generalize to non-squared.
        B, T, F, C = x.shape
        # vip B as tf.shape()

        # N_poly is length of flattened fmap after subsampling. N_poly = int(T*F / 4)
        # Original implementation assumes squared fmaps. If we do MP2x2 (ie subsampling by 2), we have (H/2 * W/2) = (N/2 * N/2) = N**2 / 4.
        # N_poly = int(N ** 2 / 4)
        # here we have T and F which are different.
        N_poly = int(T*F / 4)
        # si voy a hacer MP2x2, al final tienes H/2 y W/2. Nb2 es el final fmap size (asumiendo que es squared)
        T_sub = int(T / 2)
        F_sub = int(F / 2)

        if permute_indices is None:
            permute_indices = permute_polyphase(T, F)
            # permute_indices is the 1-D tensor (One single row) containing the concat of the 4 possible grids to index the fmap.
            # length is 4* number of elements in subsampled fmap

        # reshape by flattening time x freq as the subsampling is going to be done in 1D
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, -1, C])
        # print('reshape B, -1, C', x.shape)

        # Downsample the fmap using the 4 possible grids.
        # The subsampling grid is discarding every second bin, hence the outcome is of length N_poly = int(T*F / 4), for each of the 4 candidates.
        # x = torch.index_select(x, dim=1, index=permute_indices).view(B, 4, N_poly, C)
        x = tf.gather(params=x, indices=permute_indices, axis=-2)
        x = tf.reshape(x, [B, 4, N_poly, C])
        # print('four candidate fmaps B, 4, N_poly, C', x.shape)

        # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        # index (IntTensor or LongTensor) – the 1-D tensor containing the indices to index
        # once we have the 4 subsampled versions, reshape to (B, C, 4, N_poly) to split the 4 candidate fmaps
        # N_poly = H/2 x W/2
        # [B, 4, N_poly, C]

        if polyphase_indices is None:
            # here
            polyphase_indices = get_polyphase_indices_v2(x, apspool_criterion)
        # polyphase_indices is a vector of batch_size numbers from 0 to 3 [0, 3, 2, 3, 1, ...]
        # basically, which candidate fmap is the best, for every example in the batch
        # print('polyphase_indices must be B', polyphase_indices.shape)

        # [B, 4, N_poly, C]
        # simply select the best candidate fmap and reshape to original shape
        # output = x[batch_indices, polyphase_indices, :, :].view(B, T_sub, F_sub, C)
        # NO: need tf.gather_nd to select multiple dimensions
        # output = x[:, polyphase_indices, :, :].view(B, T_sub, F_sub, C)

        # we have to pair the selection:
        # B index - best fmap
        # 0 - 3
        # 1 - 2
        # 2 - 3
        # 3 - 0
        # ...

        # create index of batch elements to index them
        batch_indices = tf.range(B)
        batch_indices = tf.expand_dims(batch_indices, axis=1)
        # print('batch_indices must be B, 1', batch_indices.shape)
        best_fmap_per_example = tf.expand_dims(polyphase_indices, axis=1)
        # print('best_fmap_per_example must be B, 1', best_fmap_per_example.shape)

        indices_pairs = tf.concat([batch_indices, best_fmap_per_example], axis=1)
        # vip prepare shape for gather_nd. Not intuitive but correct, based on https://riptutorial.com/tensorflow/example/29069/how-to-use-tf-gather-nd
        indices = tf.expand_dims(indices_pairs, axis=1)
        out_selected_fmap = tf.gather_nd(x, indices)
        # (B, 1, N_poly, C)
        # print('indices_pairs must be B, 2', indices_pairs.shape)
        # print('indices must be B, 1, 2', indices.shape)
        # print('out_selected_fmap must be (B, 1, N_poly, C)', out_selected_fmap.shape)

        output = tf.reshape(out_selected_fmap, [B, T_sub, F_sub, C])
        # print('output must be (B, T_sub, F_sub, C)', output.shape)

        return output


def get_polyphase_indices_v2(x, apspool_criterion):
    # vip original code: x has the form (B, 4, C, N_poly) where N_poly corresponds to the reduced version of the 2d feature maps
    # vip now in TF: [B, 4, N_poly, C]
    # batch, 4 candidate fmaps, flattened fmap of length N_poly = int(T*F / 4) for each of the 4 candidates, channels
    # so the only thing that changes is the ordering of last two channels

    B, Ncandi, N_poly, C = x.shape
    B = tf.shape(x)[0]

    if apspool_criterion == 'l2':
        # axis?? 2-tuple of Python integers: axis determines the axes in tensor over which to compute a matrix norm

        # se cuelga for some reason
        # norms = tf.norm(x, ord=2, axis=(2, 3))
        # polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

        # l2 norm after flattening dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        norms = tf.norm(x, ord=2, axis=2)
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

        # original:
        # maybe compute l2 across C and Npoly, for each of the 4 candidate fmaps, hence 4 norms
        # choose the best norm, for each element in the batch, so it is a vector of batch_size numbers [0, 3, 2, 3, 1, ...]
        # norms = torch.norm(x, dim=(2, 3), p=2)
        # polyphase_indices = torch.argmax(norms, dim=1)

    elif apspool_criterion == 'l1':
        norms = tf.norm(x, ord=1, axis=(2, 3))
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

        # original: (has the same syntax because the only thing that changes is the ordering of last two channels)
        # norms = torch.norm(x, dim=(2, 3), p=1)
        # polyphase_indices = torch.argmax(norms, dim=1)

    elif apspool_criterion == 'l_infty':
        # now: [B, 4, N_poly, C]

        B = tf.shape(x)[0]
        # flatten last two dims (the flattened fmap and the depth channels)
        x = tf.reshape(x, [B, 4, -1])
        # abs value
        x = tf.math.abs(x)
        # l_inf: pick max(abs(.))
        max_vals = tf.math.reduce_max(x, axis=2)
        # (B, 4)
        polyphase_indices = tf.math.argmax(max_vals, axis=1, output_type=tf.dtypes.int32)
        # a vector of batch_size numbers [0, 3, 2, 3, 1, ...]

        # original:
        # B = x.shape[0]
        # max_vals = torch.max(x.reshape(B, 4, -1).abs(), dim=2).values
        # polyphase_indices = torch.argmax(max_vals, dim=1)

    # ========================================================================= new trials for v2 paper
    elif apspool_criterion == 'euclidean':
        # watch tenia escrito: norms = tf.norm(x, ord='euclidean', axis=(2, 3)) performs worse
        norms = tf.norm(x, ord='euclidean', axis=(2, 3))
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l_infty_tf':
        norms = tf.norm(x, ord=np.inf, axis=(2, 3))
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'var':
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 variances)
        _, variances = tf.nn.moments(x, axes=[2], name='variance')
        # (B, 4)
        # return index of candidate maximizing criterion
        polyphase_indices = tf.math.argmax(variances, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l0': # ok
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 L0 norms)
        norms = tf.math.count_nonzero(x, axis=2, dtype=tf.dtypes.int32)
        # (B, 4)
        # return index of candidate maximizing criterion
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'kurtosis': # ok
        # https://github.com/deepchem/deepchem/blob/692a2ed74a622c2beeb58494c37b553a8a98e3d2/contrib/tensorflow_models/utils.py#L152

        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])

        # compute criterion for each candidate (4 kurtosis values)
        # Compute kurtosis, the fourth standardized moment minus three (which corresponds to the kurtosis of a normal distribution)
        # reduction_indices: Axes to reduce across. If None, reduce to a scalar.
        kurtosis = Moment(4, x, standardize=True, reduction_indices=[2])[1] - 3
        # (B, 4)

        # return index of candidate maximizing criterion
        polyphase_indices = tf.math.argmax(kurtosis, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'skewness': # ok
        # https://github.com/deepchem/deepchem/blob/692a2ed74a622c2beeb58494c37b553a8a98e3d2/contrib/tensorflow_models/utils.py#L152

        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])

        # compute criterion for each candidate (4 skewness values)
        # Compute skewness, the third standardized moment.
        #     reduction_indices: Axes to reduce across. If None, reduce to a scalar.
        skewness = Moment(3, x, standardize=True, reduction_indices=[2])[1]
        # (B, 4)

        # return index of candidate maximizing criterion
        polyphase_indices = tf.math.argmax(skewness, axis=1, output_type=tf.dtypes.int32)

        # low priority==================================
    elif apspool_criterion == 'entropy':
        # fmap values will be a  lot of 0s (due to relu), then real valued positives (which may almost never repeat)
        # hence the entropy should be correlated with the number of 0 values, or, the number of non-zero values, ie the L0 norm
        # there may be some differences in certain specific cases, such as when num values = 0 is half of the total cases
        # but these cases should happen rarely and we are interested in raking the metric (not the concrete values)
        # so probably behaviour of entropy is similar to L0
        # watch an alternative:
        # do quantification of values into N groups:
        # 2 groups: values=0, and values>0. Maybe similar to L0norm?
        # watch N>2: values=0, and then split the dynamic range of the positives in N-1 bins, hence we have repeated codes, more sense for entropy

        a=0
        # https://stackoverflow.com/questions/56306943/how-to-calculate-entropy-on-float-numbers-over-a-tensor-in-python-keras
        # polyphase_indices = tf.math.argmax(entropies, axis=1, output_type=tf.dtypes.int32)


    # ============================================ original, never tried.
    # elif apspool_criterion == 'non_abs_max':
    #     B = x.shape[0]
    #     max_vals = torch.max(x.reshape(B, 4, -1), dim=2).values
    #     polyphase_indices = torch.argmax(max_vals, dim=1)
    #
    # elif apspool_criterion == 'l2_min':
    #     norms = torch.norm(x, dim=(2, 3), p=2)
    #     polyphase_indices = torch.argmin(norms, dim=1)
    #
    # elif apspool_criterion == 'l1_min':
    #     norms = torch.norm(x, dim=(2, 3), p=1)
    #     polyphase_indices = torch.argmin(norms, dim=1)

    else:
        raise Exception('Unknown APS criterion')

    return polyphase_indices


def construct_1d_array(filt_size):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    return a


def aps_pad(x):
    """
    this func pads in order to have even fmap
    reflect at the end of time and in the HF end, which are presumably less critical locations.
    :param x:
    :return:
    """

    # we cannot do circular in tf, so we do reflect y pista.
    # N1 = h is axis=-2,
    # N2 = w is axis=-1
    # N1, N2 = x.shape[2:4]
    B, T, F, C = x.shape

    if T % 2 == 0 and F % 2 == 0:
        # if even
        return x

    # pad to make even fmap
    if T % 2 != 0:
        # N1 = h is axis=-2,
        # x = tf.pad(x, (0, 0, 0, 1), mode='circular')
        x = tf.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]], mode="REFLECT")

    if F % 2 != 0:
        # N2 = w is axis=-1
        # x = tf.pad(x, (0, 1, 0, 0), mode='circular')
        x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0]], mode="REFLECT")
    # print('end aps_pad', x.shape)
    return x


def permute_polyphase(T, F, stride=2):
    """
    given the dims of a squared fmap, T and F, define the 4 possible grids to subsample the fmap with stride 2
    considering a fmap of 2D flattened out to 1D
    """

    # F
    # 2 * [0 1 2 3 4] = [0 2 4 6 8] esto son matrix or vector?
    base_even_ind = 2 * tf.range(int(F / 2))[None, :]  # ok
    # 1 + 2 * [0 1 2 3 4] = [ 1 3 5 7 9]
    base_odd_ind = 1 + 2 * tf.range(int(F / 2))[None, :] # ok

    # T
    # 20 * [0 1 2 3 4] = [0 20 40 60 80]
    # even_increment = 2 * T * tf.range(int(T / 2))[:, None]
    even_increment = 2 * F * tf.range(int(T / 2))[:, None] # ok
    # one index every second time frame. Each index must accumulate the number of freq bands so far (n2F)
    # tensor([[ 0],
    #     [20],
    #     [40],
    #     [60],
    #     [80]])

    # 30 * [0 1 2 3 4] = [10 30 50 70 90]????
    # odd_increment = T + 2 * T * tf.range(int(T / 2))[:, None]
    odd_increment = F + 2 * F * tf.range(int(T / 2))[:, None] # ok
    # one index every second time frame. Each index must accumulate the number of freq bands so far (n2F + F)
    # tensor([[10],
    #     [30],
    #     [50],
    #     [70],
    #     [90]])

    p0_indices = tf.reshape(base_even_ind + even_increment, [-1])  # flatten
    p1_indices = tf.reshape(base_even_ind + odd_increment, [-1])  # flatten
    p2_indices = tf.reshape(base_odd_ind + even_increment, [-1])  # flatten
    p3_indices = tf.reshape(base_odd_ind + odd_increment, [-1])  # flatten

    # p0_indices = (base_even_ind + even_increment).view(-1)
    # tensor([ 0,  2,  4,  6,  8, 20, 22, 24, 26, 28, 40, 42, 44, 46, 48, 60, 62, 64, 66, 68, 80, 82, 84, 86, 88])
    # p1_indices = (base_even_ind + odd_increment).view(-1)
    # tensor([10, 12, 14, 16, 18, 30, 32, 34, 36, 38, 50, 52, 54, 56, 58, 70, 72, 74, 76, 78, 90, 92, 94, 96, 98])

    # p2_indices = (base_odd_ind + even_increment).view(-1)
    # tensor([ 1,  3,  5,  7,  9, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49, 61, 63, 65, 67, 69, 81, 83, 85, 87, 89])
    # p3_indices = (base_odd_ind + odd_increment).view(-1)

    # al final, las 4 posibles sets de indices para subsamplear el fmap de manera diferente, pero en flattened to 1D
    permute_indices = tf.concat([p0_indices, p1_indices, p2_indices, p3_indices], axis=0)
    # permute_indices is the 1-D tensor (One single row) containing the concat of the 4 grids to index the fmap.
    # length is 4* number of elements in subsampled fmap
    return permute_indices



def Moment(k, tensor, standardize=False, reduction_indices=None, mask=None):
    """Compute the k-th central moment of a tensor, possibly standardized.
    Args:
    k: Which moment to compute. 1 = mean, 2 = variance, etc.
    tensor: Input tensor.
    standardize: If True, returns the standardized moment, i.e. the central
      moment divided by the n-th power of the standard deviation.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.
    Returns:
    The mean and the requested moment.
    """
    # vip reduction_indices must be a list of ints, eg [2]
    # if reduction_indices is not None:
    #     reduction_indices = np.atleast_1d(reduction_indices).tolist()

    # get the divisor
    if reduction_indices is None:
        divisor = tf.constant(np.prod(tensor.get_shape().as_list()), tensor.dtype)
    else:
        divisor = 1.0
        for i in range(len(tensor.get_shape())):
            if i in reduction_indices:
                divisor *= tensor.get_shape()[i]
        divisor = tf.constant(divisor, tensor.dtype)

    # compute the requested central moment
    # note that mean is a raw moment, not a central moment
    mean = tf.math.divide(
        tf.reduce_sum(tensor, axis=reduction_indices, keepdims=True), divisor)
    delta = tensor - mean

    moment = tf.math.divide(tf.reduce_sum(tf.math.pow(delta, k), axis=reduction_indices, keepdims=True), divisor)
    moment = tf.squeeze(moment, reduction_indices)
    if standardize:
        moment = tf.multiply(
            moment,
            tf.math.pow(
                tf.math.rsqrt(Moment(2, tensor, reduction_indices=reduction_indices)[1]),
                k))

    return tf.squeeze(mean, reduction_indices), moment


