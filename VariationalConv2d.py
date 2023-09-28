from keras_core import activations, initializers, regularizers, ops, random
from keras_core.layers import Layer

class VariationalConv2d(Layer):
    def __init__(self, kernel_size, stride, padding='SAME', threshold=3.0, activation=None, data_format='channels_last', kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=None):
        super(VariationalConv2d, self).__init__()
        assert len(kernel_size) == 4    #kernel_size: [filter_height, filter_width, in_channels, out_channels]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.activation = activations.get(activation)
        if data_format == 'channels_first':
            self.data_format = 'NCHW'
        else:
            self.data_format = 'NHWC'

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.theta = self.add_weight("kernel", shape=self.kernel_size,
                                     initializer=self.kernel_initializer, trainable=True, regularizer=self.kernel_regularizer)

        self.log_sigma2 = self.add_weight("log_sigma2", shape=self.kernel_size,
                                 initializer=initializers.Constant(-10.0), trainable=True)

    def sparsity(self):
        total_param = ops.prod(ops.shape(self.boolean_mask))
        remaining_param = ops.count_nonzero(ops.cast(self.boolean_mask, dtype="uint8"))

        return remaining_param, total_param

    @property
    def log_alpha(self):
        theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)
        log_sigma2 = ops.where(ops.isnan(self.log_sigma2), ops.zeros_like(self.log_sigma2), self.log_sigma2)
        log_alpha = ops.clip(log_sigma2 - ops.log(ops.square(theta) + 1e-10), -20.0, 4.0)
        return ops.where(ops.isnan(log_alpha), self.threshold * ops.ones_like(log_alpha), log_alpha)

    @property
    def boolean_mask(self):
        return self.log_alpha <= self.threshold

    @property
    def sparse_theta(self):
        theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)
        return ops.where(self.boolean_mask, theta, ops.zeros_like(theta))

    @property
    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * ops.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * ops.log(1 + (ops.exp(-self.log_alpha))) + C

        return -ops.sum(mdkl)


    def call(self, input, sparse=False):
        theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)

        if not sparse:
            sigma = ops.sqrt(ops.exp(self.log_alpha) * theta * theta)
            self.weight = theta + random.normal(ops.shape(theta), 0.0, 1.0) * sigma
            output = ops.conv(input, self.weight, self.stride, self.padding, self.data_format)

            if self.activation is not None:
                output = self.activation(output)

            return output

        else:
            output = ops.conv(input, self.sparse_theta, self.stride, self.padding, self.data_format)
            if self.activation is not None:
                output = self.activation(output)

            return output


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "activation": activations.serialize(self.activation),
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "threshold": self.threshold,
            }
        )
        return config


