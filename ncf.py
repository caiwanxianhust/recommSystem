import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class NCF(object):
    def __init__(self, num_users, num_items, emb_size=16, drop_rate=.2):
        self.user_inp = keras.Input(shape=(1,), name="user_inp")
        self.item_inp = keras.Input(shape=(1,), name="item_inp")
        self.user_embedding_gmf = keras.layers.Embedding(num_users + 1, emb_size, name="user_emb_gmf")
        self.item_embedding_gmf = keras.layers.Embedding(num_items + 1, emb_size, name="item_emb_gmf")
        self.user_embedding_mlp = keras.layers.Embedding(num_users + 1, emb_size, name="user_emb_mlp")
        self.item_embedding_mlp = keras.layers.Embedding(num_items + 1, emb_size, name="item_emb_mlp")
        self.multiply_gmf = keras.layers.Multiply(name="multiply_gmf")
        self.concate_emb = keras.layers.Concatenate(axis=-1, name="concate_emb")
        self.reshape_layer = keras.layers.Reshape((emb_size,), name="reshape")
        self.mlp_1 = keras.layers.Dense(emb_size * 2, activation="relu")
        self.mlp_2 = keras.layers.Dense(emb_size, activation="relu")
        self.mlp_3 = keras.layers.Dense(emb_size, activation="relu")
        self.dropout_1 = keras.layers.Dropout(rate=drop_rate)
        self.dropout_2 = keras.layers.Dropout(rate=drop_rate)
        self.dropout_3 = keras.layers.Dropout(rate=drop_rate)
        self.concate_com = keras.layers.Concatenate(axis=-1, name="concate_com")
        self.com_dense = keras.layers.Dense(1, activation="sigmoid", name="com_dense")

    def build(self):
        # (batch_size, 1) -> (batch_size, 1, emb_size)
        user_emb_gmf = self.user_embedding_gmf(self.user_inp)
        # (batch_size, 1, emb_size) -> (batch_size, emb_size)
        user_emb_gmf = self.reshape_layer(user_emb_gmf)
        # (batch_size, 1) -> (batch_size, 1, emb_size)
        item_emb_gmf = self.item_embedding_gmf(self.item_inp)
        # (batch_size, 1, emb_size) -> (batch_size, emb_size)
        item_emb_gmf = self.reshape_layer(item_emb_gmf)
        # (batch_size, emb_size)
        self.gmf_out = self.multiply_gmf([user_emb_gmf, item_emb_gmf])
        # (batch_size, 1) -> (batch_size, 1, emb_size)
        user_emb_mlp = self.user_embedding_mlp(self.user_inp)
        # (batch_size, 1, emb_size) -> (batch_size, emb_size)
        user_emb_mlp = self.reshape_layer(user_emb_mlp)
        # (batch_size, 1) -> (batch_size, 1, emb_size)
        item_emb_mlp = self.item_embedding_mlp(self.item_inp)
        # (batch_size, 1, emb_size) -> (batch_size, emb_size)
        item_emb_mlp = self.reshape_layer(item_emb_mlp)
        # (batch_size, emb_size) + (batch_size, emb_size) -> (batch_size, emb_size * 2)
        interaction = self.concate_emb([user_emb_mlp, item_emb_mlp])
        # (batch_size, emb_size * 2) -> (batch_size, emb_size * 2)
        mlp_out = self.mlp_1(interaction)
        mlp_out = self.dropout_1(mlp_out)
        # (batch_size, emb_size * 2) -> (batch_size, emb_size)
        mlp_out = self.mlp_2(mlp_out)
        mlp_out = self.dropout_2(mlp_out)
        # (batch_size, emb_size) -> (batch_size, emb_size)
        mlp_out = self.mlp_3(mlp_out)
        self.mlp_out = self.dropout_3(mlp_out)
        # (batch_size, emb_size) + (batch_size, emb_size) -> (batch_size, emb_size * 2)
        self.com_out = self.concate_com([self.gmf_out, self.mlp_out])
        # (batch_size, emb_size * 2) -> (batch_size, 1)
        self.com_out = self.com_dense(self.com_out)
        self.gmf_model = keras.Model([self.user_inp, self.item_inp], self.gmf_out, name="gmf")
        self.mlp_model = keras.Model([self.user_inp, self.item_inp], self.mlp_out, name="mlp")
        self.neumf_model = keras.Model([self.user_inp, self.item_inp], self.com_out, name="neumf")
        self.neumf_model.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer="adam",
                                 metrics=[keras.metrics.binary_accuracy, keras.metrics.Recall()])


if __name__ == "__main__":
    ncfal = NCF(5000, 7000)
    ncfal.build()
    ncfal.neumf_model.summary()
    keras.utils.plot_model(ncfal.gmf_model, 'ncf_gmf_model.png', show_layer_names=True, show_shapes=True)
    keras.utils.plot_model(ncfal.mlp_model, 'ncf_mlp_model.png', show_layer_names=True, show_shapes=True)
    keras.utils.plot_model(ncfal.neumf_model, 'ncf_neumf_model.png', show_layer_names=True, show_shapes=True)
