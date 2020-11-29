import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class InteractionLayer(keras.layers.Layer):
    def __init__(self, factor_dim=16, **kwargs):
        self.factor_dim = factor_dim
        super(InteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inp, **kwargs):
        # (v_if * x_i)^2 (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        a = tf.reduce_sum(K.pow(K.dot(inp, self.kernel), 2), axis=1, keepdims=True)
        # (v_if^2 * x_i) (batch_size, n_features）->(batch_size, k) ->(batch, 1)
        b = tf.reduce_sum(K.dot(inp ** 2, self.kernel ** 2), axis=1, keepdims=True)
        interaction_out = a - b
        # (batch_size, n_features） -> (batch_size, n_features * k）
        rep_inp = K.repeat_elements(inp, rep=self.factor_dim, axis=1)
        # (n_features, k) -> (1, n_features * k)
        flatten_kernel = tf.reshape(self.kernel, shape=(1, -1))
        # (batch_size, n_features * k)
        flatten_inp_emb = rep_inp * flatten_kernel
        return (interaction_out, flatten_inp_emb)


class DeepFM:
    def __init__(self, n_features, factor_dim):
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.lr = keras.layers.Dense(1, use_bias=True, name="lr")
        self.interaction = InteractionLayer(factor_dim, name="interaction")
        self.concate_inp_layer = keras.layers.Concatenate(axis=-1, name="concate_inp_layer")
        self.concate_out_layer = keras.layers.Concatenate(axis=-1, name="concate_out_layer")
        self.com_dense = keras.layers.Dense(1, activation="sigmoid")

    def build(self):
        # (batch_size, n_features) -> (batch_size, 1)
        self.lr_out = self.lr(self.inp)
        # (batch_size, n_features) -> (batch_size, 1)
        self.interaction_out, self.flatten_inp_emb = self.interaction(self.inp)
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.flatten_inp_emb))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu')(self.deep_out))
        self.deep_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1, activation='relu')(self.deep_out))
        self.out = self.concate_out_layer([self.lr_out, self.interaction_out, self.deep_out])
        self.out = self.com_dense(self.out)
        self.model = keras.Model(self.inp, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["acc"])


if __name__ == "__main__":
    deepfm = DeepFM(31, 8)
    deepfm.build()
    deepfm.model.summary()
    keras.utils.plot_model(deepfm.model, "deepfm.png", show_layer_names=True, show_shapes=True)
