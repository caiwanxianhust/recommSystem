import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class CateEmbedding(keras.layers.Layer):
    def __init__(self, emb_dim, **kwargs):
        self.emb_dim = emb_dim
        super(CateEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape: field个数
        :return:
        """
        self.kernel = self.add_weight(name='cate_em_vecs',
                                      shape=(input_shape[1], self.emb_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        x = K.expand_dims(x, axis=2)
        x = K.repeat_elements(x, rep=self.emb_dim, axis=2)
        out = x * self.kernel
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.emb_dim)


class PairWiseInteraction(keras.layers.Layer):
    def __init__(self, mask=True, **kwargs):
        self.mask = mask
        super(PairWiseInteraction, self).__init__(**kwargs)

    def call(self, x):
        """
        pair-wise interaction layer
        :param x: (batch_size, n_features, emb_dim)
        :return:
        """
        # (batch_size, n_features, emb_dim) -> (batch_size, n_features, 1, emb_dim)
        x = K.expand_dims(tf.cast(x, tf.float32), axis=2)
        # (batch_size, n_features, 1, emb_dim) -> (batch_size, n_features, n_features, emb_dim)
        x = K.repeat_elements(x, rep=x.shape[1], axis=2)
        xt = tf.transpose(x, perm=[0, 2, 1, 3])
        out = x * xt
        if self.mask:
            # (1, emb_dim, n_features, n_features)
            mask = 1 - tf.linalg.band_part(tf.ones((1, out.shape[3], out.shape[1], out.shape[2])), -1, 0)
            # (1, n_features, n_features, emb_dim)
            mask = tf.transpose(mask, perm=[0, 2, 3, 1])
            out = out * mask
        return tf.reshape(out, shape=(-1, out.shape[1] * out.shape[2], out.shape[3]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[1], input_shape[2])


class AttentionPSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.p_sum_layer = keras.layers.Dense(1, use_bias=False)
        super(AttentionPSum, self).__init__(**kwargs)

    def call(self, inp):
        """

        :param inp: attention_out:(batch_size, n_features * n_features, 1);
                    wise_product:(batch_size, n_features * n_features, emb_dim)
        :return:
        """
        attention_out, wise_product = inp
        # (batch_size, emb_dim)
        out = tf.reduce_sum(tf.multiply(attention_out, wise_product), axis=1)
        # (batch_size, 1)
        out = self.p_sum_layer(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], 1)


class AttentionFM(object):
    def __init__(self, n_features, emb_dim=16, attention_dim=16):
        self.inp = keras.Input(shape=(n_features,), name="inp")
        self.embedding_layer = CateEmbedding(emb_dim, name="embedding_layer")
        self.lr_layer = keras.layers.Dense(1, use_bias=True, name="lr")
        self.wise_product_layer = PairWiseInteraction(mask=True, name="wise_product_interaction")
        self.attention_wb = keras.layers.Dense(attention_dim, use_bias=True, name="attention_wb")
        self.attention_h = keras.layers.Dense(1, use_bias=False, name="attention_h")
        self.attention_softmax = keras.layers.Lambda(lambda x: K.softmax(x, axis=1), name="attention_softmax")
        self.attention_psum = AttentionPSum(name="psum")
        self.add_lr_ap = keras.layers.Add()
        self.sigmoid = keras.layers.Activation("sigmoid")

    def build(self):
        # (batch_size, n_features) -> (batch_size, 1)
        lr = self.lr_layer(self.inp)
        # (batch_size, n_features) -> (batch_size, n_features, emb_dim)
        x = self.embedding_layer(self.inp)
        # (batch_size, n_features * n_features, emb_dim)
        wise_product = self.wise_product_layer(x)
        # (batch_size, n_features * n_features, attention_size)
        a_out = self.attention_wb(wise_product)
        # (batch_size, n_features * n_features, 1)
        a_out = self.attention_h(a_out)
        # (batch_size, n_features * n_features, 1)
        a_out = self.attention_softmax(a_out)
        # (batch_size, n_features * n_features, emb_dim) -> (batch_size, emb_dim) ->(batch_size, 1)
        p_sum = self.attention_psum([a_out, wise_product])
        self.out = self.add_lr_ap([lr, p_sum])
        self.out = self.sigmoid(self.out)
        self.model = keras.Model(self.inp, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer="adam",
                           metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()])


if __name__ == "__main__":
    afmal = AttentionFM(40)
    afmal.build()
    afmal.model.summary()
