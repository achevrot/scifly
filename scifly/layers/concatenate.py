import tensorflow as tf

class Concatenate(tf.keras.layers.Layer):
    
    def __init__(self, name="concatenate", **kwargs):
        super(Concatenate, self).__init__(name=name, **kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=0)
    
    def call(self, inputs):
        asc_dec, cru_dec, des_dec, concat_index = inputs
        concat_dec = self.concat([asc_dec, cru_dec, des_dec])
        return tf.gather(concat_dec, tf.argsort(concat_index), axis=0)

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config.update({"trainable": False})
        return config 
