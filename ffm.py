import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class FFMLayer(keras.layers.Layer):
    def __init__(self, factor_dim, field_dict, **kwargs):
        self.factor_dim = factor_dim
        self.field_dict = field_dict
        self.n_features = len(field_dict)
        self.field_dim = len(set(field_dict.values()))
        super(FFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.n_features, self.field_dim, self.factor_dim),
                                 trainable=True,
                                 initializer='random_uniform')

        super(FFMLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        interaction_term = tf.zeros(shape=(1,), dtype=tf.float32)
        for i in range(self.n_features - 1):
            for j in range(i + 1, self.n_features):
                wij = tf.reduce_sum(
                    tf.math.multiply(self.w[i, self.field_dict[j], :], self.w[j, self.field_dict[i], :]))
                interaction_term += wij * tf.math.multiply(x[:, i], x[:, j])
        interaction_term = tf.reshape(interaction_term, [-1, 1])
        return interaction_term


class FieldawareFactorizationMachine:
    def __init__(self, factor_dim, field_dict, n_features):
        self.factor_dim = factor_dim
        self.field_dict = field_dict
        self.x_in = keras.Input(shape=(n_features,), name="inp")
        self.lr = keras.layers.Dense(1, use_bias=True)
        self.ffm_layer = FFMLayer(factor_dim, field_dict)
        self.activate = keras.layers.Activation("sigmoid")

    def build(self):
        lr_term = self.lr(self.x_in)
        interaction_term = self.ffm_layer(self.x_in)
        self.out = lr_term + interaction_term
        self.out = self.activate(self.out)
        self.model = keras.Model(self.x_in, self.out)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer='adam',
                           metrics=['acc'])


if __name__ == "__main__":
    field_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6:4, 7:4, 8:4, 9:5, 10:6, 11:7}

    ffmal = FieldawareFactorizationMachine(8, field_dict, 12)
    ffmal.build()
    ffmal.model.summary()
