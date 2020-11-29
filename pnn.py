import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pandas as pd


class CateEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, **kwargs):
        self.emb_dim = emb_dim
        super(CateEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='emb_vecs',
                                      shape=(1, input_shape[1] * self.emb_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        x = K.repeat_elements(x, rep=self.emb_dim, axis=1)
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.emb_dim)


class InnerQuadratic(keras.layers.Layer):
    def __init__(self, deep_init_size, **kwargs):
        super(InnerQuadratic, self).__init__(**kwargs)
        self.deep_init_size = deep_init_size

    def build(self, input_shape):
        self.kernel = self.add_weight(name='inner_quadratic',
                                      shape=(input_shape[1], self.deep_init_size),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x):
        # (batch_size, field_size, embedding_size)
        out = tf.matmul(x, self.kernel, transpose_a=True, transpose_b=True)
        out = tf.norm(out, axis=1)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.deep_init_size)


class OuterQuadratic(keras.layers.Layer):
    def __init__(self, deep_init_size, **kwargs):
        super(OuterQuadratic, self).__init__(**kwargs)
        self.deep_init_size = deep_init_size
        self.dense = keras.layers.Dense(deep_init_size)

    def build(self, input_shape):
        pass

    def call(self, x):
        # x :(batch_size, field_size, embedding_size)
        embedding_sum = tf.reduce_sum(x, axis=1)  # (batch_size, embedding_size)
        p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))
        p = tf.reshape(p, (-1, embedding_size * embedding_size))
        out = self.dense(p)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.deep_init_size)


class InteractionLayer(keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def call(self, x):
        embedding_sum = tf.reduce_sum(x, axis=1)  # (batch_size, embedding_size)
        # (batch_size, embedding_size, embedding_size)
        p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))
        # (batch_size, embedding_size * embedding_size)
        p = tf.reshape(p, (-1, self.embedding_size * self.embedding_size))
        return p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_size * self.embedding_size)


class PNN(object):
    def __init__(self, field_size, embedding_size, product_units, num_deep_layers, deep_layer_units=8, use_inner=True):
        self.product_units = product_units
        self.field_size = field_size
        self.embedding_layer = CateEmbedding(embedding_size)
        self.linear_signal = keras.layers.Dense(product_units, use_bias=True, name="linear")
        self.reshape_layer = keras.layers.Reshape((field_size, embedding_size), name="reshape_layer")
        self.use_inner = use_inner
        self.interaction = InteractionLayer(embedding_size)
        self.deep_layers = [keras.layers.Dense(deep_layer_units, use_bias=True,
                                               activation="relu") for _ in range(num_deep_layers)]
        self.final_dense = keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="out")
        self.x_in = keras.Input(shape=(field_size, ), name="inp")

    def build(self):
        # (batch_size, field_size) -> (batch_size, field_size*embedding_size)
        x = self.embedding_layer(self.x_in)
        # Linear Singal   (batch_size, field_size*embedding_size) -> (batch_size, product_units)
        lz = self.linear_signal(x)
        # (batch_size, field_size*embedding_size) -> (batch_size, field_size, embedding_size)
        reshape_x = self.reshape_layer(x)
        if self.use_inner:
            # (batch_size, field_size, embedding_size) -> (batch_size, embedding_size, field_size)
            reshape_transpose_x = keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]),
                                                      name="transpose")(reshape_x)
            # (batch_size, embedding_size, field_size) ->  (batch_size, embedding_size, product_units)
            lp = keras.layers.Dense(self.product_units)(reshape_transpose_x)
            # (batch_size, embedding_size, product_units) -> (batch_size, product_units)
            lp = keras.layers.Lambda(lambda x: tf.norm(x, axis=1), name="norm")(lp)
        else:
            # (batch_size, embedding_size * embedding_size)
            p = self.interaction(reshape_x)
            lp = keras.layers.Dense(self.product_units)(p)
        y_deep = keras.layers.Add()([lz, lp])
        y_deep = keras.layers.Activation(activation="relu")(y_deep)
        y_deep = keras.layers.Dropout(rate=.3)(y_deep)
        for layer in self.deep_layers:
            y_deep = layer(y_deep)
            y_deep = keras.layers.Dropout(rate=.2)(y_deep)
        self.out = self.final_dense(y_deep)
        self.model = keras.Model(self.x_in, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam",
                           metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()])


if __name__ == "__main__":
    pnnal = PNN(40, 16, 16, 3, use_inner=False)
    pnnal.build()
    pnnal.model.summary()