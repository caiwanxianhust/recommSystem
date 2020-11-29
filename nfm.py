import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras


class BiInteractionLayer(keras.layers.Layer):
    def __init__(self, factor_dim=16, **kwargs):
        self.factor_dim = factor_dim
        super(BiInteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inp):
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        interaction_out = a - b
        return interaction_out


class NFM:
    def __init__(self, n_features, factor_dim):
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.lr = keras.layers.Dense(1, use_bias=True, name="lr")
        self.interaction = BiInteractionLayer(factor_dim, name="bi_interaction")
        self.add_out_layer = keras.layers.Add(name="add_out_layer")

    def build(self):
        # (batch_size, n_features) -> (batch_size, 1)
        self.lr_out = self.lr(self.inp)
        # (batch_size, n_features) -> (batch_size, 1)
        self.interaction_out = self.interaction(self.inp)
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.interaction_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1, activation='relu')(self.deep_out))
        self.out = self.add_out_layer([self.lr_out, self.deep_out])
        self.model = keras.Model(self.inp, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["acc"])


if __name__ == "__main__":
    n_features, factor_dim = 40, 16
    nfmal = NFM(n_features, factor_dim)
    nfmal.build()
    nfmal.model.summary()
