import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'jax'
#os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np

from src.sparse_vd_keras.VariationalDense import VariationalDense
from src.sparse_vd_keras.VariationalConv2d import VariationalConv2d
from sklearn.utils import shuffle

from keras import Model, ops, utils, datasets, losses, optimizers, metrics, Variable
from keras.layers import MaxPooling2D, Flatten

def rw_schedule(epoch):
    """Defines the schedule for the weight regularization term."""
    return ops.where(epoch <= 1.0, 0.0, 0.0001 * (epoch - 1.0))


class VariationalLeNet(Model):
    def __init__(self, n_class=10):
        super().__init__()
        self.n_class = n_class

        self.conv1 = VariationalConv2d((5, 5, 1, 6), stride=1, padding='valid')
        self.pooling1 = MaxPooling2D(padding='same')
        self.conv2 = VariationalConv2d((5, 5, 6, 16), stride=1, padding='valid')
        self.pooling2 = MaxPooling2D(padding='same')

        self.flat = Flatten()
        self.fc1 = VariationalDense(120)
        self.fc2 = VariationalDense(84)
        self.fc3 = VariationalDense(10)

        self.hidden_layers = [self.conv1, self.pooling1, self.conv2, self.pooling2, self.flat, self.fc1, self.fc2, self.fc3]

        self.epoch = Variable(initializer=0.0, dtype='float32', trainable=False)

    def call(self, input, **kwargs):
        x = self.conv1(input, sparse_input=kwargs['sparse_input'])
        x = ops.relu(x)
        x = self.pooling1(x)
        x = self.conv2(x, sparse_input=kwargs['sparse_input'])
        x = ops.relu(x)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x, sparse_input=kwargs['sparse_input'])
        x = ops.relu(x)
        x = self.fc2(x, sparse_input=kwargs['sparse_input'])
        x = ops.relu(x)
        x = self.fc3(x, sparse_input=kwargs['sparse_input'])
        x = ops.softmax(x)
        self.epoch.assign_add(1.0)
        return x

    def build(self, input_shape):
        for layer in self.hidden_layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    @property
    def regularization(self):
        """Computes the total regularization term that has been applied on all the layers."""
        total_reg = 0
        for layer in self.hidden_layers:
            try:
                total_reg += layer.regularization
            except AttributeError:
                continue

        return total_reg

    @property
    def count_sparsity(self):
        """Computes the sparsity of the weight matrices in all the layers."""
        total_remain, total_param = 0, 0
        for layer in self.hidden_layers:
            try:
                a, b = layer.sparsity()
            except AttributeError:
                continue
            total_remain += a
            total_param += b

        return 1 - (total_remain / total_param)


if __name__ == '__main__':
    utils.set_random_seed(1234)

    '''
    Load data
    '''
    mnist = datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = VariationalLeNet()
    criterion = losses.CategoricalCrossentropy()
    optimizer = optimizers.AdamW()

    '''
    Train model
    '''
    epochs = 20
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    if os.environ['KERAS_BACKEND'] != 'jax':
        train_loss = metrics.Mean()
        test_loss = metrics.Mean()

    train_acc = metrics.CategoricalAccuracy()
    test_acc = metrics.CategoricalAccuracy()

    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        import tensorflow as tf

        @tf.function
        def compute_loss(label, pred, reg):
            return criterion(label, pred) + reg

        @tf.function
        def compute_loss2(label, pred):
            return criterion(label, pred)

        @tf.function
        def train_step(x, t):
            with tf.GradientTape() as tape:
                preds = model(x, train=True, sparse_input=False)
                reg = rw_schedule(model.epoch.value) * model.regularization()
                loss = compute_loss(t, preds, reg)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)
            train_acc.update_state(t, preds)

            return preds

        @tf.function
        def test_step(x, t):
            preds = model(x, train=False, sparse_input=True)
            loss = compute_loss2(t, preds)
            test_loss(loss)
            test_acc.update_state(t, preds)

            return preds


        for epoch in range(epochs):
            _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                train_step(ops.convert_to_tensor(_x_train[start:end], dtype="float32"),
                           ops.convert_to_tensor(_y_train[start:end], dtype="float32"))

            # Epoch valiation
            preds = test_step(ops.convert_to_tensor(x_test, dtype="float32"),
                              ops.convert_to_tensor(y_test, dtype="float32"))
            print(f'Epoch: {epoch + 1}, Valid Cost: {test_loss.result():.3f}, Valid Acc: {test_acc.result():.3f}')
            print("Sparsity: ", ops.convert_to_numpy(model.count_sparsity()))


    elif os.environ['KERAS_BACKEND'] == 'jax':

        import jax

        def compute_loss(t, preds, reg):
            return criterion(t, preds) + reg

        def compute_loss2(label, pred):
            return criterion(label, pred)

        def compute_loss_and_updates(trainable_variables, non_trainable_variables, metric_variables, x, t, epoch,
                                     training=False):
            reg = rw_schedule(epoch) * model.regularization()

            preds, non_trainable_variables = model.stateless_call(trainable_variables, non_trainable_variables, x,
                                                                  train=training, sparse_input=False)

            loss = compute_loss(t, preds, reg)
            metric_variables = train_acc.stateless_update_state(metric_variables, t, preds)
            return loss, (non_trainable_variables, metric_variables)

        grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

        @jax.jit
        def train_step(state, data):
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metric_variables,
            ) = state
            x, t = data

            (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(trainable_variables,
                                                                                 non_trainable_variables,
                                                                                 metric_variables,
                                                                                 x, t, model.epoch.value, True)

            trainable_variables, optimizer_variables = optimizer.stateless_apply(optimizer_variables, grads,
                                                                                 trainable_variables)

            # Return updated state
            return loss, (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metric_variables,
            )

        @jax.jit
        def eval_step(state, data):
            trainable_variables, non_trainable_variables, optimizer_variables, metric_variables = state
            x, t = data

            preds, non_trainable_variables = model.stateless_call(trainable_variables, non_trainable_variables,
                                                                  x, train=False, sparse_input=True)

            loss = compute_loss2(t, preds)

            metric_variables = test_acc.stateless_update_state(metric_variables, t, preds)

            return loss, (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metric_variables,
            )


        # Initialization
        model.build(x_train.shape)
        optimizer.build(model.trainable_variables)
        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        optimizer_variables = optimizer.variables
        metric_variables = train_acc.variables
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metric_variables,
        )

        for epoch in range(epochs):
            x_train, y_train = shuffle(x_train, y_train, random_state=42)

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                loss, state = train_step(state,
                                         (ops.convert_to_tensor(x_batch, dtype="float32"),
                                          ops.convert_to_tensor(y_batch, dtype="float32")))

            loss, state = eval_step(state,
                                    (ops.convert_to_tensor(x_test, dtype="float32"),
                                     ops.convert_to_tensor(y_test, dtype="float32")))
            for variable, value in zip(train_acc.variables, metric_variables):
                variable.assign(value)

            print(f'Epoch: {epoch + 1}, Valid Cost: {float(loss):.3f}, Valid Acc: {test_acc.result():.3f}')
            print("Sparsity: ", ops.convert_to_numpy(model.count_sparsity()))


    elif os.environ['KERAS_BACKEND'] == 'torch':
        import torch

        def compute_loss(label, pred, reg):
            return criterion(label, pred) + reg

        def compute_loss2(label, pred):
            return criterion(label, pred)

        def train_step(x, t):
            preds = model(x, train=True, sparse_input=False)
            reg = rw_schedule(model.epoch.value) * model.regularization()
            loss = compute_loss(t, preds, reg)

            model.zero_grad()
            trainable_weights = [v for v in model.trainable_weights]

            loss.backward()
            gradients = [v.value.grad for v in trainable_weights]

            # Since a Keras optimizer is used, the gradients are applied to the trainable weights manually, rather than using optimizer.step()
            with torch.no_grad():
                optimizer.apply(gradients, trainable_weights)

            train_loss(loss.item())
            train_acc.update_state(t, preds)
            return preds

        def test_step(x, t):
            preds = model(x, train=False, sparse_input=True)
            loss = compute_loss2(t, preds)
            test_loss(loss.item())
            test_acc.update_state(t, preds)
            return preds


        for epoch in range(epochs):
            _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                x_batch = ops.convert_to_tensor(_x_train[start:end], dtype="float32")
                y_batch = ops.convert_to_tensor(_y_train[start:end], dtype="float32")
                train_step(x_batch, y_batch)

            x_test_tensor = ops.convert_to_tensor(x_test, dtype="float32")
            y_test_tensor = ops.convert_to_tensor(y_test, dtype="float32")
            preds = test_step(x_test_tensor, y_test_tensor)
            print(f'Epoch: {epoch + 1}, Valid Cost: {test_loss.result():.3f}, Valid Acc: {test_acc.result():.3f}')
            print("Sparsity: ", ops.convert_to_numpy(model.count_sparsity()))
