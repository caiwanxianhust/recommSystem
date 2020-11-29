import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class FMLayer(keras.layers.Layer):
    def __init__(self, k=16, activation="sigmoid", **kwargs):
        self.lr = keras.layers.Dense(1, use_bias=True)
        self.k = k
        self.activate = keras.layers.Activation(activation)
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                        shape=(input_shape[1], self.k),
                        initializer='glorot_uniform',
                        trainable=True)

    def call(self, inp, **kwargs):
        # 线性模型部分（batch_size, n_features）->(batch_size, 1)
        lr = self.lr(inp)
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        out = lr + a - b
        out = self.activate(out)
        return out


class FM(object):
    def __init__(self, k, n_features):
        self.fm = FMLayer(k)
        self.x_in = keras.Input(shape=(n_features,), name="inp")

    def build(self):
        self.out = self.fm(self.x_in)
        self.model = keras.Model(self.x_in, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, metrics=["acc"], optimizer="adam")

