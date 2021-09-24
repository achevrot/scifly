#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:57:52 2021

@author: antoine
"""

import tensorflow as tf
from tensorflow.keras.layers import GRU, RepeatVector, Dense, Bidirectional, TimeDistributed, Flatten, LSTM
import datetime
from vincenty import vincenty

# %%

tf.random.set_seed(3333)

params = {}
params["BATCH_SIZE"] = 256
params["TIME_STEPS"] = 15
params["FEATURE_NUMBERS"] = 4
params["LATENT_DIM"] = 10

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
        [v for k, v in sequence.items() if k in ('alt', 'lat', 'lon', 'spd')], dtype=tf.float32))

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
    
    # lat_min = tf.math.minimum(departure_coordinate[0], arrival_coordinate[0])
    # lat_max = tf.math.maximum(departure_coordinate[0], arrival_coordinate[0])
    # lon_min = tf.math.minimum(departure_coordinate[1], arrival_coordinate[1])
    # lon_max = tf.math.maximum(departure_coordinate[1], arrival_coordinate[1])
    lat_min = tf.reduce_min(sequence['lat'])
    lat_max = tf.reduce_max(sequence['lat'])
    lon_min = tf.reduce_min(sequence['lon'])
    lon_max = tf.reduce_max(sequence['lon'])

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
                                     sizes=[1, params["TIME_STEPS"], 1, 1],
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
        
    dataset = (dataset.unbatch().
               batch(params['BATCH_SIZE'], drop_remainder=True).
               shuffle(params['BATCH_SIZE']*2).prefetch(1))
    dataset = dataset.map(lambda x: (tf.ensure_shape(x, [None, params["TIME_STEPS"], params["FEATURE_NUMBERS"]])))
    return dataset


# %%

class Sampling(tf.keras.layers.Layer):

    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""


    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder_inputs  = tf.keras.Input(batch_size=params["BATCH_SIZE"], shape=(
    params["TIME_STEPS"], params["FEATURE_NUMBERS"]), name='features', dtype=tf.float32)

e = Bidirectional(LSTM(20, return_sequences=True))(encoder_inputs)
e = Flatten()(e)
z_mean = Dense(params["LATENT_DIM"], name='z_mean')(e)
z_log_var = Dense(params["LATENT_DIM"], name='z_log_var')(e)
z = Sampling()(z_mean, z_log_var)

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(params["LATENT_DIM"],), name='z_sampling')
# latent_inputs = tf.keras.Input(shape=(params["TIME_STEPS"],40), name='z_sampling')
repeat = RepeatVector(params["TIME_STEPS"])(latent_inputs)
d = Bidirectional(LSTM(20, return_sequences=True))(repeat)
outputs = Dense(params["FEATURE_NUMBERS"], activation='softplus')(d)

decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

z_mean, z_log_var, z = encoder(encoder_inputs)
dec_outputs = decoder(z)
vae = tf.keras.Model(inputs=encoder_inputs, outputs=dec_outputs, name="vae")



# Add KL divergence regularization loss.

def kl_divergence(X, X_pred):
    kl_loss = -0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_sum(kl_loss, axis=1)
    return tf.reduce_mean(kl_loss)

def reconstruction_loss(X, X_pred):
    mse = tf.keras.losses.MSE
    return tf.reduce_mean(tf.reduce_sum(mse(X, X_pred), axis=1))

def loss(X, X_pred):
    return reconstruction_loss(X, X_pred) + 0.1*kl_divergence(X, X_pred)

  
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
vae.compile(optimizer='adam', loss=loss, metrics=[kl_divergence, reconstruction_loss])
# %%

ds_train = data_preparation("../../../../Data/Training_data/autoencoder/*tfrecord")
ds_val = data_preparation("../../../../Data/Validation_data/autoencoder/*tfrecord")
ds_train = ds_train.map(lambda x: (x, x))
ds_val = ds_val.map(lambda x: (x, x))

logdir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_filepath = '../tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_reconstruction_loss',
    mode='min',
    save_best_only=True)


# vae.fit(ds_train.take(200),validation_data=ds_val.take(5), epochs=60)
vae.fit(ds_train, validation_data=ds_val, epochs=100, verbose = 2, callbacks=[
                                                    model_checkpoint_callback,
                                                    tensorboard_callback])

vae.save('../model')



#%%

model = tf.keras.models.load_model('../tmp/checkpoint', compile=False)


