#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:57:52 2021

@author: antoine
"""

import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional, Flatten
import datetime
from scifly.layers import PhaseSplitter
from vincenty import vincenty

# %%

tf.random.set_seed(1234)


params = {}
params["BATCH_SIZE"] = 512
params["TIME_STEPS"] = 30
params["FEATURE_NUMBERS"] = 10
params["LATENT_DIM"] = 20

# %%

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 batch_size=params['BATCH_SIZE'],
                 timesteps=params['TIME_STEPS'],
                 n_features=params["FEATURE_NUMBERS"],
                 latent_dim=params["LATENT_DIM"],
                 intermediate_dim=32,
                 name="encoder", **kwargs):

        super(Encoder, self).__init__(name=name, **kwargs)
        self.lstm = LSTM(intermediate_dim, return_sequences=True)
        self.flatten = Flatten()
        self.out = Dense(latent_dim)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.flatten(x)
        return self.out(x)

    def get_config(self):
        config = super(Encoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 batch_size=params['BATCH_SIZE'],
                 timesteps=params['TIME_STEPS'],
                 n_features=params["FEATURE_NUMBERS"],
                 latent_dim=params["LATENT_DIM"],
                 intermediate_dim=32, name="decoder", **kwargs):

        super(Decoder, self).__init__(name=name, **kwargs)
        self.repeat = RepeatVector(timesteps)
        self.lstm = LSTM(intermediate_dim, return_sequences=True)
        self.dense = Dense(n_features)

    def call(self, inputs):
        x = self.repeat(inputs)
        x = self.lstm(x)
        return self.dense(x)

    def get_config(self):
        config = super(Decoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# %%


def _parse_function(example_proto):

    # Define features
    context_features = {
        'icao': tf.io.FixedLenFeature([], dtype=tf.string),
        'cal': tf.io.FixedLenFeature([], dtype=tf.string),
        'dep_airport_lat': tf.io.FixedLenFeature([], dtype=tf.float32),
        'dep_airport_lon': tf.io.FixedLenFeature([], dtype=tf.float32),
        'arr_airport_lat': tf.io.FixedLenFeature([], dtype=tf.float32),
        'arr_airport_lon': tf.io.FixedLenFeature([], dtype=tf.float32)
    }

    sequence_features = {
        'alt': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'label' : tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'lat': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'lon': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'vert_rate': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'hdg': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'spd': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'd_a': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'd_b': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'd_c': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'd_d': tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    # Parsing the records
    context, sequence = tf.io.parse_single_sequence_example(
        example_proto, context_features, sequence_features)

    departure_coordinate = (context['dep_airport_lat'], context['dep_airport_lon'])
    arrival_coordinate = (context['arr_airport_lat'], context['arr_airport_lon'])
    normalize_sequence(sequence, departure_coordinate, arrival_coordinate)
    
    ds_features = tf.transpose(tf.cast(
        [v for k, v in sequence.items() if k not in ('label', 'phase', 'vert_rate')], dtype=tf.float32))

    return ds_features

def tf_vincenty(departure_coordinate, arrival_coordinate):
  flight_length = tf.py_function(
    vincenty,
    (departure_coordinate,arrival_coordinate),  # pass these args to the above function.
    tf.float32)      # the return type is `tf.string`.
  return tf.reshape(flight_length, ()) # The result is a scalar


def normalize_sequence(sequence, departure_coordinate, arrival_coordinate):
    
    flight_length = tf_vincenty(departure_coordinate, arrival_coordinate)
    sequence['alt'] = sequence['alt'] / 40000
    
    lat_min = tf.math.minimum(departure_coordinate[0], arrival_coordinate[0])
    lat_max = tf.math.maximum(departure_coordinate[0], arrival_coordinate[0])
    lon_min = tf.math.minimum(departure_coordinate[1], arrival_coordinate[1])
    lon_max = tf.math.maximum(departure_coordinate[1], arrival_coordinate[1])

    sequence['lat'] = (sequence['lat'] - lat_min) / (lat_max - lat_min)
    sequence['lon'] = (sequence['lon'] - lon_min) / (lon_max - lon_min)
    sequence['hdg'] = sequence['hdg'] / 360
    sequence['spd'] = sequence['spd'] / 500
    sequence['delta'] = sequence['delta'] / 10
    sequence['d_a'] = sequence['d_a'] / flight_length
    sequence['d_b'] = sequence['d_b'] / flight_length
    sequence['d_c'] = sequence['d_c'] / flight_length
    sequence['d_d'] = sequence['d_d'] / flight_length
    
def tensor_windowing(x):
    a = tf.reshape(x, [1, -1, x.shape[1], 1])
    a_win = tf.image.extract_patches(a,
                                     sizes=[1, 30, 1, 1],
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='VALID')[0, :, :]
    a_win = tf.transpose(a_win, perm=[0, 2, 1])
    return a_win


def data_preparation(path):

    files = tf.data.Dataset.list_files([path])
    raw_dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    dataset = dataset.map(lambda x: (tensor_windowing(x)))
    dataset = dataset.unbatch().shuffle(params['BATCH_SIZE']*10).batch(
        params['BATCH_SIZE'], drop_remainder=True).prefetch(1)
    dataset = dataset.map(lambda x: (tf.ensure_shape(x, [None, params["TIME_STEPS"], params["FEATURE_NUMBERS"]])))
    return dataset

# %%


inputs = tf.keras.Input(shape=(
    params["TIME_STEPS"], params["FEATURE_NUMBERS"]), name='features', dtype=tf.float32)
encoded_features = Encoder()(inputs)
decoder = Decoder(name="asc_dec")(encoded_features)

# %%

model = tf.keras.Model(inputs=inputs,
                       outputs=decoder, name='autoencoder')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=optimizer, loss='mse')

ds_train = data_preparation("../../../../Data/Training_data/autoencoder/*tfrecord")
ds_val = data_preparation("../../../../Data/Validation_data/autoencoder/*tfrecord")
ds_train = ds_train.map(lambda x: (x, x))
ds_val = ds_val.map(lambda x: (x, x))

logdir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_filepath = '../tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


model.fit(ds_train, validation_data=ds_val, epochs=100, callbacks=[
                                                    model_checkpoint_callback,
                                                    tensorboard_callback])
model.save('./habler')
