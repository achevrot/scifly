#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:57:52 2021

@author: antoine
"""

import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Flatten, RepeatVector
from scifly.layers import PhaseSplitter
import datetime
#%%

tf.random.set_seed(1234)

params = {}
params["BATCH_SIZE"] = 256
params["TIME_STEPS"] = 30
params["FEATURE_NUMBERS"] = 5
params["LATENT_DIM"] = 10

#%%


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

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 batch_size=params['BATCH_SIZE'],
                 timesteps=params['TIME_STEPS'],
                 n_features=params["FEATURE_NUMBERS"],
                 latent_dim=params["LATENT_DIM"],
                 intermediate_dim=32,
                 name="encoder", **kwargs):
        
        super(Encoder, self).__init__(name=name, **kwargs)
        self.bidirectional = Bidirectional(LSTM(intermediate_dim, return_sequences=True),
        backward_layer=LSTM(intermediate_dim, return_sequences=True, go_backwards=True))
        self.flatten = Flatten()
        self.out = Dense(latent_dim)

    def call(self, inputs):
        e = self.bidirectional(inputs)
        e = self.flatten(e)
        return self.out(e)
    
    
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
        
    @tf.function    
    def call(self, inputs):
        if tf.equal(tf.size(inputs), 0):
            return tf.zeros([0, params['TIME_STEPS'], params["FEATURE_NUMBERS"]])
        d = self.repeat(inputs)
        d = self.lstm(d)
        return self.dense(d)
    
    def get_config(self):
      config = super(Decoder, self).get_config()
      return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    'phase': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
    'track_delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'spd': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    'vert_rate': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        }
    
    # Parsing the records
    context, sequence = tf.io.parse_single_sequence_example(
        example_proto, context_features, sequence_features)
    
    ds_features = tf.transpose(tf.cast([v for k,v in sequence.items() if k not in ('label','phase')], dtype=tf.float32))
    label = tf.cast(sequence['label'], dtype=tf.int64)
    phase = tf.cast(sequence['phase'], dtype=tf.string)
    
    return ds_features, phase

def get_norm_layer(path):
    files = tf.data.Dataset.list_files([path])
    raw_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    phase = dataset.map(lambda x,y:y)
    phase_norm_layer = get_category_encoding_layer(phase, 'string')
    
    features = dataset.map(lambda a,b:a)
    feature_norm_layer = get_normalization_layer(features)
    return phase_norm_layer, feature_norm_layer


def tensor_windowing(x):
    a = tf.reshape(x,[1,-1,x.shape[1],1])
    a_win = tf.image.extract_patches(a,
                        sizes=[1, params["TIME_STEPS"], 1, 1],
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='VALID')[0,:,:]
    a_win = tf.transpose(a_win, perm=[0,2,1])
    return a_win

def data_preparation(path, phase_norm_layer, feature_norm_layer):

    files = tf.data.Dataset.list_files([path])
    raw_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    result_ds = dataset.map(lambda x, y: (tensor_windowing(feature_norm_layer(x)), tensor_windowing(phase_norm_layer(tf.reshape(y,[-1,1])))))
    ds = result_ds.unbatch().batch(params['BATCH_SIZE'], drop_remainder=True).shuffle(params['BATCH_SIZE']).prefetch(1)
    ds = ds.map(lambda x, y: (tf.ensure_shape(x,[None,params["TIME_STEPS"],params["FEATURE_NUMBERS"]]),tf.ensure_shape(tf.cast(tf.reshape(tf.reduce_max(y, axis=1),[-1]),tf.float32),[params['BATCH_SIZE'],])))
    return ds

# %%

inputs = tf.keras.Input(shape=(params["TIME_STEPS"], params["FEATURE_NUMBERS"]), name='features', dtype=tf.float32)
phases = tf.keras.Input(shape=(1,), name='phases', dtype=tf.float32)

encoded_features = Encoder()(inputs)
asc_data, cru_data, des_data, concat_index = PhaseSplitter()((encoded_features, phases))

asc_dec = Decoder(name="asc_dec")(asc_data)
cru_dec = Decoder(name="cru_dec")(cru_data)
des_dec = Decoder(name="des_dec")(des_data)

# outputs = Concatenate()((asc_dec, cru_dec, des_dec, concat_index))

concat = tf.keras.layers.Concatenate(axis=0)([asc_dec, cru_dec, des_dec])
outputs = tf.gather(concat, tf.argsort(concat_index), axis=0)

# %%

model = tf.keras.Model(inputs=(inputs,phases), outputs=outputs, name='autoencoder')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=optimizer, loss='mse')

phase_norm_layer, feature_norm_layer = get_norm_layer("../../../../Data/Training_data/multi_ae/*.tfrecord")
ds_train = data_preparation("../../../../Data/Training_data/multi_ae/*.tfrecord", phase_norm_layer, feature_norm_layer)
ds_val = data_preparation("../../../../Data/Validation_data/multi_ae/*tfrecord", phase_norm_layer, feature_norm_layer)
ds_train = ds_train.map(lambda x,y:((x,y),x))
ds_val = ds_val.map(lambda x,y:((x,y),x))

logdir="../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_filepath = '../tmp/checkpoint2'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# model.fit(ds_train, validation_data=ds_val,epochs=100)

model.fit(ds_train, validation_data=ds_val, epochs=100, callbacks=[
                                                    model_checkpoint_callback,
                                                    tensorboard_callback])

model.save('../model_17_06')
    
