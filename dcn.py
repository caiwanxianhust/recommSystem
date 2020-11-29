import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class CateEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, **kwargs):
        self.emb_dim = emb_dim
        super(CateEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='cate_em_vecs',
                                      shape=(1, input_shape[1] * self.emb_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        x = K.repeat_elements(x, rep=self.emb_dim, axis=1)
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.emb_dim)


class CrossLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.cross_dense = keras.layers.Dense(1, use_bias=True)
        super(CrossLayer, self).__init__(**kwargs)

    def call(self, inp, **kwargs):
        x0, xl = inp
        if (K.ndim(x0) <= 2):
            x0 = x0[..., tf.newaxis]
        if (K.ndim(xl) <= 2):
            xl = xl[..., tf.newaxis]
        # [batch_size, n_features, 1] -> [batch_size, n_features, n_features]
        x0l = tf.matmul(x0, xl, transpose_b=True)
        out = self.cross_dense(x0l) + xl
        out = tf.reshape(out, (-1, out.shape[1]))
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])


class DeepCrossNetwork:
    def __init__(self, n_cate_features, n_numeric_features, emb_dim, num_cross_layers, dnn_units=[8, 8, 8], rate=.2):
        self.cate_inp = keras.Input(shape=(n_cate_features,), name="cate_inp")
        self.numeric_inp = keras.Input(shape=(n_numeric_features,), name="numeric_inp")
        self.cate_emb_layer = CateEmbedding(emb_dim, name="cate_emb")
        self.concate_inp = keras.layers.Concatenate(axis=-1, name="concate_inp")
        self.cross_layers = [CrossLayer(name="cross_{}".format(i)) for i in range(num_cross_layers)]
        # self.reshape = keras.layers.Reshape((n_cate_features + n_numeric_features,))
        self.dnn_layers = [keras.layers.Dense(units, activation="relu") for units in dnn_units]
        self.rate = rate
        self.concate_out = keras.layers.Concatenate(axis=-1, name="concate_out")
        self.com_dense = keras.layers.Dense(1, activation="sigmoid")

    def build(self):
        # (batch_size, n_cate_features) -> (batch_size, n_cate_features * emb_dim)
        cate_emb = self.cate_emb_layer(self.cate_inp)
        # (batch_size, n_cate_features * emb_dim + n_numeric_featrues)
        inp = self.concate_inp([cate_emb, self.numeric_inp])
        xl = inp
        for layer in self.cross_layers:
            xl = layer([inp, xl])
        y_deep = keras.layers.Dropout(self.rate)(inp)
        for layer in self.dnn_layers:
            y_deep = layer(y_deep)
            y_deep = keras.layers.Dropout(self.rate)(y_deep)
        self.combine_out = self.concate_out([xl, y_deep])
        self.combine_out = self.com_dense(self.combine_out)
        self.model = keras.Model([self.cate_inp, self.numeric_inp], self.combine_out)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer="adam",
                           metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()])


if __name__ == "__main__":
    dcn = DeepCrossNetwork(37, 3, 16, 3)
    dcn.build()
    dcn.model.summary()
    keras.utils.plot_model(dcn.model, "dcn.png", show_layer_names=True, show_shapes=True)
