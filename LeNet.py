import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['KERAS_BACKEND'] = 'jax'
#os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np

from VariationalDense import VariationalDense
from VariationalConv2d import VariationalConv2d
from sklearn.utils import shuffle

from keras_core import Model, ops, utils, datasets, losses, optimizers, metrics
from keras_core.layers import MaxPooling2D, Flatten

def rw_schedule(epoch):
    if epoch <= 1:
        return 0
    else:
        return 0.0001 * (epoch - 1)


class VariationalLeNet(Model):
    def __init__(self, n_class=10):
        super().__init__()
        self.n_class = n_class

        self.conv1 = VariationalConv2d((5, 5, 1, 6), stride=1, padding='VALID')
        self.pooling1 = MaxPooling2D(padding='SAME')
        self.conv2 = VariationalConv2d((5, 5, 6, 16), stride=1, padding='VALID')
        self.pooling2 = MaxPooling2D(padding='SAME')

        self.flat = Flatten()
        self.fc1 = VariationalDense(120)
        self.fc2 = VariationalDense(84)
        self.fc3 = VariationalDense(10)

        self.hidden_layer = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def call(self, x, sparse=False):
        x = self.conv1(x, sparse)
        x = ops.relu(x)
        x = self.pooling1(x)
        x = self.conv2(x, sparse)
        x = ops.relu(x)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x, sparse)
        x = ops.relu(x)
        x = self.fc2(x, sparse)
        x = ops.relu(x)
        x = self.fc3(x, sparse)
        x = ops.softmax(x)

        return x

    def regularization(self):
        total_reg = 0
        for layer in self.hidden_layer:
            total_reg += layer.regularization

        return total_reg

    def count_sparsity(self):
        total_remain, total_param = 0, 0
        for layer in self.hidden_layer:
            a, b = layer.sparsity()
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

    train_loss = metrics.Mean()
    train_acc = metrics.CategoricalAccuracy()
    test_loss = metrics.Mean()
    test_acc = metrics.CategoricalAccuracy()

    if os.environ['KERAS_BACKEND'] == 'tensorflow':
        @tf.function
        def compute_loss(label, pred, reg):
            return criterion(label, pred) + reg

        @tf.function
        def compute_loss2(label, pred):
            return criterion(label, pred)

        @tf.function
        def train_step(x, t, epoch):
            with tf.GradientTape() as tape:
                preds = model(x)
                reg = rw_schedule(epoch) * model.regularization()
                loss = compute_loss(t, preds, reg)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)
            train_acc(t, preds)

            return preds

        @tf.function
        def test_step(x, t):
            preds = model(x, sparse=True)
            loss = compute_loss2(t, preds)
            test_loss(loss)
            test_acc(t, preds)

            return preds


        for epoch in range(epochs):

            _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                train_step(_x_train[start:end], _y_train[start:end], epoch)

            if epoch % 1 == 0 or epoch == epochs - 1:
                preds = test_step(x_test, y_test)
                print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
                    epoch + 1,
                    test_loss.result(),
                    test_acc.result()
                ))
                print("Sparsity: ", model.count_sparsity())

    elif os.environ['KERAS_BACKEND'] == 'jax':
        def compute_loss(trainable_variables, non_trainable_variables, metric_variables, label, reg):
            pred, non_trainable_variables = model.stateless_call(
                trainable_variables, non_trainable_variables, label
            )
            loss = criterion(label, pred) + reg
            metric_variables = train_acc.stateless_update_state(
                metric_variables, label, pred
            )
            return loss, (non_trainable_variables, metric_variables)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

        def compute_loss2(trainable_variables, non_trainable_variables, metric_variables, label):
            pred, non_trainable_variables = model.stateless_call(
                trainable_variables, non_trainable_variables, label
            )
            loss = criterion(label, pred)
            metric_variables = train_acc.stateless_update_state(
                metric_variables, label, pred
            )
            return loss, (non_trainable_variables, metric_variables)

        @jax.jit
        def train_step(state, data):
            (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables) = state
            label = data
            reg = rw_schedule(epoch) * model.regularization()
            (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
                trainable_variables, non_trainable_variables, metric_variables, label, reg
            )
            trainable_variables, optimizer_variables = optimizer.stateless_apply(
                optimizer_variables, trainable_variables, grads
            )
            # Return updated state
            return loss, (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metric_variables
            )


        @jax.jit
        def eval_step(state, data):
            trainable_variables, non_trainable_variables, metric_variables = state
            label = data
            pred, non_trainable_variables = model.stateless_call(
                trainable_variables, non_trainable_variables, label
            )
            loss = compute_loss2(label, pred)
            metric_variables = val_acc_metric.stateless_update_state(
                metric_variables, label, pred
            )
            return loss, (
                trainable_variables,
                non_trainable_variables,
                metric_variables,
            )


        optimizer.build(model.trainable_variables)

        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        optimizer_variables = optimizer.variables
        metric_variables = train_acc.variables
        state = (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables)

        for step, data in enumerate(train_dataset):
            data = (data[0].numpy(), data[1].numpy())
            loss, state = train_step(state, data)
            # Log every 100 batches.
            if step % 100 == 0:
                print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")