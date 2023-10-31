from keras_core import activations, initializers, regularizers, ops, random
from keras_core.layers import Layer
from keras_core import saving

@saving.register_keras_serializable(package="VariationalDropoutAutoencoder")
class VariationalDense(Layer):
    def __init__(self, output_dim, use_bias=True, threshold=3.0, activation=None, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=None):
        super(VariationalDense, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.threshold = threshold
        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.seed_generator = random.SeedGenerator()

    def build(self, input_shape):
        self.theta = self.add_weight(name="kernel", shape=(int(input_shape[-1]), self.output_dim),
                                     initializer=self.kernel_initializer, trainable=True, regularizer=self.kernel_regularizer)

        self.log_sigma2 = self.add_weight(name="log_sigma2", shape=(int(input_shape[-1]), self.output_dim),
                                 initializer=initializers.Constant(-10.0), trainable=True)

        if self.use_bias == True:
            self.bias = self.add_weight(name="bias", shape=(self.output_dim,),
                                        initializer=self.bias_initializer, trainable=True)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def sparsity(self):
        """Compute sparsity of the weight matrix, based on the number of non-zero elements."""
        total_param = ops.prod(ops.array(ops.shape(self.boolean_mask)))
        remaining_param = ops.count_nonzero(ops.cast(self.boolean_mask, dtype="uint8"))

        return remaining_param, total_param

    @property
    def log_alpha(self):
        """Compute log alpha, which is the log of the variance of the weight matrix."""
        theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
        log_sigma2 = ops.where(ops.isnan(self.log_sigma2.value), ops.zeros_like(self.log_sigma2.value), self.log_sigma2.value)
        log_alpha = ops.clip(log_sigma2 - ops.log(ops.square(theta) + 1e-10), -20.0, 4.0)
        return ops.where(ops.isnan(log_alpha), self.threshold * ops.ones_like(log_alpha), log_alpha)


    @property
    def boolean_mask(self):
        return self.log_alpha <= self.threshold

    @property
    def sparse_theta(self):
        theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
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
        if not kwargs['sparse_input']:
            theta = ops.where(ops.isnan(self.theta.value), ops.zeros_like(self.theta.value), self.theta.value)
            sigma = ops.sqrt(ops.exp(self.log_alpha) * theta * theta)
            self.weight = theta + random.normal(ops.shape(theta), 0.0, 1.0, seed=self.seed_generator) * sigma
            output = ops.matmul(input, self.weight)
            if self.use_bias == True:
                output += self.bias
            if self.activation is not None:
                output = self.activation(output)

        else:
            output = ops.matmul(input, self.sparse_theta)
            if self.use_bias == True:
                output += self.bias
            if self.activation is not None:
                output = self.activation(output)

        return output

    def get_config(self):
        base_config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "threshold": self.threshold,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return {**base_config, **config}



if __name__ == '__main__':
    a = VariationalDense(10)
    print(a(model(ops.zeros((1, 1)), sparse_input=True)))
    print(a.regularization)






