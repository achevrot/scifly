import tensorflow as tf
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Dense

class Decoder(tf.keras.layers.Layer):
    def __init__(self, timesteps, n_features, latent_dim=5,
                 intermediate_dim=15, name="encoder", **kwargs):
        
        super(Decoder, self).__init__(name=name, **kwargs)
        self.repeat = RepeatVector(timesteps)
        self.time_distributed = TimeDistributed(Dense(n_features))

    def call(self, inputs):
        x = self.repeat(inputs)
        return self.time_distributed(x)

    def get_config(self):
        config = super(Decoder, self).get_config()
        return config
