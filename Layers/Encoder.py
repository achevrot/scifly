import tensorflow as tf
from tensorflow.keras.layers import LSTM

class Encoder(tf.keras.layers.Layer):
    def __init__(self, timesteps, n_features, latent_dim=5,
                 intermediate_dim=15, name="encoder", **kwargs):
        
        super(Encoder, self).__init__(name=name, **kwargs)
        self.lstm = LSTM(intermediate_dim,
                         input_shape=[timesteps, n_features],
                         return_sequences=True)
        self.lstm_2 = LSTM(latent_dim)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.lstm_2(x)

    def get_config(self):
        config = super(Encoder, self).get_config()
        return config
