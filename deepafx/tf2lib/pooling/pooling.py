# MIT License
# 
# Copyright (c) 2018 Brian McFee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import regularizers

class AutoPool1D(Layer):
    '''Automatically tuned soft-max pooling.
    This layer automatically adapts the pooling behavior to interpolate
    between mean- and max-pooling for each dimension.
    '''
    def __init__(self, axis=0,
                 kernel_initializer='zeros',
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 **kwargs):
        '''
        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0 (should be time).
        kernel_initializer: Initializer for the weights matrix
        kernel_regularizer: Regularizer function applied to the weights matrix
        kernel_constraint: Constraint function applied to the weights matrix
        kwargs
        '''

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )

        super(AutoPool1D, self).__init__(**kwargs)

        self.axis = axis
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(1, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def get_config(self):
        config = {'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'axis': self.axis}

        base_config = super(AutoPool1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        scaled = self.kernel * x
        max_val = K.max(scaled, axis=self.axis, keepdims=True)
        softmax = K.exp(scaled - max_val)
        weights = softmax / K.sum(softmax, axis=self.axis, keepdims=True)
        return K.sum(x * weights, axis=self.axis, keepdims=False)


class SoftMaxPool1D(Layer):
    '''
    Keras softmax pooling layer.
    '''

    def __init__(self, axis=0, **kwargs):
        '''
        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0
            (should be time).
        kwargs
        '''
        super(SoftMaxPool1D, self).__init__(**kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        max_val = K.max(x, axis=self.axis, keepdims=True)
        softmax = K.exp((x - max_val))
        weights = softmax / K.sum(softmax, axis=self.axis, keepdims=True)
        return K.sum(x * weights, axis=self.axis, keepdims=False)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SoftMaxPool1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))