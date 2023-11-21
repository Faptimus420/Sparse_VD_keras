from keras_core import activations, initializers, regularizers, ops, random
from keras_core.layers import Layer
from keras_core import saving
import numpy as np
import os

def compute_conv_output_shape(
    input_shape,
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    """Compute the output shape of conv ops.
    Copied and modified from https://github.com/keras-team/keras-core/blob/3a464f904dc8dd09de9c51d248a7fe7c575d7f3e/keras_core/ops/operation_utils.py,
    since the original function is not exposed."""

    if data_format == "channels_last":
        spatial_shape = input_shape[1:-1]
        kernel_shape = kernel_size + (input_shape[-1], filters)
    else:
        spatial_shape = input_shape[2:]
        kernel_shape = kernel_size + (input_shape[1], filters)

    if len(kernel_shape) != len(input_shape):
        raise ValueError(
            "Kernel shape must have the same length as input, but received "
            f"kernel of shape {kernel_shape} and "
            f"input of shape {input_shape}."
        )
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(spatial_shape)
    if isinstance(strides, int):
        strides = (strides,) * len(spatial_shape)
    if len(dilation_rate) != len(spatial_shape):
        raise ValueError(
            "Dilation must be None, scalar or tuple/list of length of "
            "inputs' spatial shape, but received "
            f"`dilation_rate={dilation_rate}` and "
            f"input of shape {input_shape}."
        )
    none_dims = []
    spatial_shape = np.array(spatial_shape)
    for i in range(len(spatial_shape)):
        if spatial_shape[i] is None:
            # Set `None` shape to a manual value so that we can run numpy
            # computation on `spatial_shape`.
            spatial_shape[i] = -1
            none_dims.append(i)

    kernel_spatial_shape = np.array(kernel_shape[:-2])
    dilation_rate = np.array(dilation_rate)
    if padding == "valid":
        output_spatial_shape = (
            np.floor(
                (spatial_shape - dilation_rate * (kernel_spatial_shape - 1) - 1)
                / strides
            )
            + 1
        )
        for i in range(len(output_spatial_shape)):
            if i not in none_dims and output_spatial_shape[i] < 0:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={input_shape}`, "
                    f"`kernel shape={kernel_shape}`, "
                    f"`dilation_rate={dilation_rate}`."
                )
    elif padding == "same" or padding == "causal":
        output_spatial_shape = np.floor((spatial_shape - 1) / strides) + 1
    output_spatial_shape = [int(i) for i in output_spatial_shape]
    for i in none_dims:
        output_spatial_shape[i] = None
    output_spatial_shape = tuple(output_spatial_shape)
    if data_format == "channels_last":
        output_shape = (
            (input_shape[0],) + output_spatial_shape + (kernel_shape[-1],)
        )
    else:
        output_shape = (input_shape[0], kernel_shape[-1]) + output_spatial_shape
    return output_shape


@saving.register_keras_serializable(package="VariationalDropoutAutoencoder", name="VariationalConv2d")
class VariationalConv2d(Layer):
    def __init__(self, kernel_size, stride, padding='same', threshold=3.0, activation=None, data_format='channels_last', kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=None, **kwargs):
        super(VariationalConv2d, self).__init__(**kwargs)
        assert len(kernel_size) == 4    #kernel_size: [filter_height, filter_width, in_channels, out_channels]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.activation = activations.get(activation)
        self.data_format = data_format

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.seed_generator = random.SeedGenerator()

    def build(self, input_shape):
        self.theta = self.add_weight(name="kernel", shape=self.kernel_size,
                                     initializer=self.kernel_initializer, trainable=True, regularizer=self.kernel_regularizer)

        self.log_sigma2 = self.add_weight(name="log_sigma2", shape=self.kernel_size,
                                 initializer=initializers.Constant(-10.0), trainable=True)

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.kernel_size[-1],
            self.kernel_size[:2],
            strides=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=1,
        )

    def sparsity(self):
        """Compute sparsity of the weight matrix, based on the number of non-zero elements."""
        total_param = ops.prod(ops.array(ops.shape(self.boolean_mask)))
        remaining_param = ops.count_nonzero(ops.cast(self.boolean_mask, dtype="uint8"))

        return remaining_param, total_param

    @property
    def log_alpha(self):
        """Compute log alpha, which is the log of the variance of the weight matrix."""
        if os.environ['KERAS_BACKEND'] == 'jax':
            theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
            log_sigma2 = ops.where(ops.isnan(self.log_sigma2.value), ops.zeros_like(self.log_sigma2.value), self.log_sigma2.value)
        else:
            theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)
            log_sigma2 = ops.where(ops.isnan(self.log_sigma2), ops.zeros_like(self.log_sigma2), self.log_sigma2)
        log_alpha = ops.clip(log_sigma2 - ops.log(ops.square(theta) + 1e-10), -20.0, 4.0)
        return ops.where(ops.isnan(log_alpha), self.threshold * ops.ones_like(log_alpha), log_alpha)

    @property
    def boolean_mask(self):
        return self.log_alpha <= self.threshold

    @property
    def sparse_theta(self):
        if os.environ['KERAS_BACKEND'] == 'jax':
            theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
        else:
            theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)
        return ops.where(self.boolean_mask, theta, ops.zeros_like(theta))

    @property
    def regularization(self):
        """Compute the regularization term for the weight matrix."""
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * ops.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * ops.log(1 + (ops.exp(-self.log_alpha))) + C

        return -ops.sum(mdkl)


    def call(self, input, **kwargs):
        """Forward pass of the layer - differentiates between sparse (on evaluation time) and dense (on training time) input."""
        if os.environ['KERAS_BACKEND'] == 'jax':
            theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
        else:
            theta = ops.where(ops.isnan(self.theta), ops.zeros_like(self.theta), self.theta)

        if not kwargs['sparse_input']:
            sigma = ops.sqrt(ops.exp(self.log_alpha) * theta * theta)
            self.weight = theta + random.normal(ops.shape(theta), 0.0, 1.0, seed=self.seed_generator) * sigma
            output = ops.conv(input, self.weight, self.stride, self.padding, self.data_format)

            if self.activation is not None:
                output = self.activation(output)

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


