import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class MixedLogisticRegression(object):
    def __init__(self, n_features, num_blocks, **kwargs):
        super(MixedLogisticRegression, self).__init__(**kwargs)
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.block_layer = keras.layers.Dense(num_blocks, use_bias=False, activation="sigmoid",
                                              kernel_regularizer='l1_l2', name="block_layer")
        self.lr = keras.layers.Dense(num_blocks, use_bias=False, activation="sigmoid",
                                     kernel_regularizer='l1_l2', name="lr")
        self.multiply_layer = keras.layers.Multiply(name="multiply_layer")
        self.sum_layer = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name="sum_layer")

    def build(self):
        # [batch_size, n_features] -> [batch_size, num_blocks]
        block_out = self.block_layer(self.inp)
        # [batch_size, n_features] -> [batch_size, num_blocks]
        lr_out = self.lr(self.inp)
        # [batch_size, n_features] -> [batch_size, num_blocks]
        self.out = self.multiply_layer([block_out, lr_out])
        # [batch_size, n_features] -> [batch_size, 1]
        self.out = self.sum_layer(self.out)
        self.mlr_model = keras.Model(self.inp, self.out)
        self.mlr_model.compile(loss=keras.losses.binary_crossentropy,
                               metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()],
                               optimizer=keras.optimizers.Ftrl())


if __name__ == "__main__":
    mlral = MixedLogisticRegression(n_features=108, num_blocks=12)
    mlral.build()
    mlral.mlr_model.summary()
