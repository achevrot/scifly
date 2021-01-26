#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:57:52 2021

@author: antoine
"""

import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import datetime
from scifly.layers import Decoder
from scifly.layers import Encoder
from scifly.layers import Concatenate
from scifly.layers import PhaseSplitter


def get_category_encoding_layer(dataset, dtype, max_tokens=4):
    
    # From https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)
    
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(dataset)
    
    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size(), output_mode="binary")
    
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(index)
    
    # Learn the space of possible indices.
    encoder.adapt(feature_ds)
    
    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: index(feature)

def get_normalization_layer(dataset):
    
    # From https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
    
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()
    # Learn the statistics of the data.
    normalizer.adapt(dataset)
    
    return normalizer

#%% 

def _parse_function(example_proto):

    # Define features
    context_features = {
        'icao': tf.io.FixedLenFeature([], dtype=tf.string),
        'cal': tf.io.FixedLenFeature([], dtype=tf.string)
        }
    
    sequence_features = {
    'alt': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'label': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'lat': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'lon': tf.io.FixedLenSequenceFeature([], dtype=tf.float32), 
    'phase': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
    'track_delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'spd': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        }

    # Parsing the records
    context, sequence = tf.io.parse_single_sequence_example(
        example_proto, context_features, sequence_features)
    
    ds_features = tf.transpose(tf.cast([v for k,v in sequence.items() if k not in ('label','phase')], dtype=tf.float32))
    phase = tf.cast(sequence['phase'], dtype=tf.string)
    
    return ds_features, phase

files = tf.data.Dataset.list_files(["/home/antoine/Documents/Test_data/training_test/*tfrecord"])
raw_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

phase = dataset.map(lambda x,y:y)
phase_norm_layer = get_category_encoding_layer(phase, 'string')

features = dataset.map(lambda a,b:a)
feature_norm_layer = get_normalization_layer(features)


def tensor_windowing(x):
    a = tf.reshape(x,[1,-1,x.shape[1],1])
    a_win = tf.image.extract_patches(a,
                        sizes=[1, 30, 1, 1],
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')[0,:,:]
    a_win = tf.transpose(a_win, perm=[0,2,1])
    return a_win

result_ds = dataset.map(lambda x, y: (tensor_windowing(feature_norm_layer(x)), tensor_windowing(phase_norm_layer(tf.reshape(y,[-1,1])))))
train_ds = result_ds.unbatch().shuffle(1024).batch(1024, drop_remainder=True).prefetch(1)
train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x,[None,30,6]),tf.ensure_shape(tf.cast(tf.reshape(tf.reduce_max(y, axis=1),[-1]),tf.float32),[1024,])))


# %%

timesteps = 30
n_features = 6



inputs = tf.keras.Input(shape=(timesteps, n_features), name='features')
phases = tf.keras.Input(shape=(1,), name='phases')

encoded_features = Encoder(timesteps,n_features)(inputs)
asc_data, cru_data, des_data, concat_index = PhaseSplitter()((encoded_features, phases))

asc_dec = Decoder(timesteps,n_features, name="asc_dec")(asc_data)
cru_dec = Decoder(timesteps,n_features, name="cru_dec")(cru_data)
des_dec = Decoder(timesteps,n_features, name="des_dec")(des_data)

outputs = Concatenate()((asc_dec, cru_dec, des_dec, concat_index))

# %%

model = tf.keras.Model(inputs=(inputs,phases), outputs=outputs, name='autoencoder')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mse')

y_train = train_ds.map(lambda x,y:((x,y),x))


logdir="/home/antoine/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


model.fit(y_train, epochs=300, callbacks=[tensorboard_callback])

# %%


# @tf.function
# def traceme(x):
#     return model(x)


# logdir = "log"
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True, profiler=True)
# # Forward pass
# traceme(next(iter(y_train)))
# with writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)


