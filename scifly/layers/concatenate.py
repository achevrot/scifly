import tensorflow as tf
from tensorflow.keras.layers import Concatenate

class Concatenate(tf.keras.layers.Layer):

    def call(self, inputs):
        asc_dec, cru_dec, des_dec, concat_index = inputs
        concat_dec = Concatenate([asc_dec, cru_dec, des_dec], 0)
        return tf.gather(concat_dec, tf.argsort(concat_index), axis=0)

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config.update({"trainable": False})
        return config 
