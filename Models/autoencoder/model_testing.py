#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:17:15 2021

@author: antoine
"""

import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import datetime
from scifly.layers import PhaseSplitter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
import configparser
from traffic.core.traffic import Traffic, Flight
from vincenty import vincenty

# import seaborn
# seaborn.set(style='ticks')

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
    
def _parse_function(example_proto):

    # Define features
    context_features = {
        'icao': tf.io.FixedLenFeature([], dtype=tf.string),
        'cal': tf.io.FixedLenFeature([], dtype=tf.string)
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
    
    ds_features = tf.transpose(tf.cast([v for k,v in sequence.items() if k not in ('label')], dtype=tf.float32))    
    return ds_features

#%%

def normalize(input_df, departure_coordinate, arrival_coordinate):
    
    df = input_df.copy()
    
    flight_length = vincenty(departure_coordinate, arrival_coordinate)
    
    
    df['altitude'] = df['altitude'] / 40000
    
    lat_min = min(departure_coordinate[0], arrival_coordinate[0])
    lat_max = max(departure_coordinate[0], arrival_coordinate[0])
    lon_min = min(departure_coordinate[1], arrival_coordinate[1])
    lon_max = max(departure_coordinate[1], arrival_coordinate[1])

    df['latitude'] = (df['latitude'] - lat_min) / (lat_max - lat_min)
    df['longitude'] = (df['longitude'] - lon_min) / (lon_max - lon_min)
    df['track'] = df['track'] / 360
    df['groundspeed'] = df['groundspeed'] / 500
    df['delta'] = df['delta'] / 10
    df['d_a'] = df['d_a'] / flight_length
    df['d_b'] = df['d_b'] / flight_length
    df['d_c'] = df['d_c'] / flight_length
    df['d_d'] = df['d_d'] / flight_length
    return df

def get_norm_layer(path):
    files = tf.data.Dataset.list_files([path])
    raw_dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    feature_norm_layer = get_normalization_layer(dataset)
    return feature_norm_layer


#%%



def get_dataset(data_raw, departure_coordinate, arrival_coordinate):
    
    flight_names=[]
    
    def parser(df):
        label = df.pop('label')
        df = normalize(df, departure_coordinate, arrival_coordinate)
        df = df[['altitude', 'd_a','d_b', 'd_c', 'd_d', 'delta', 'track', 'latitude', 'longitude', 'groundspeed']]
        df.reset_index(drop=True, inplace=True)
        
        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        dataset = dataset.window(30, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x,y : tf.data.Dataset.zip((x.batch(30, drop_remainder=True), y.batch(30, drop_remainder=True))))
        dataset = dataset.map(lambda x, y: (tf.reshape(x,[30, 10]), tf.reshape(tf.reduce_max(y, axis=0),[-1]))).batch(len(df))
        return dataset
    
    if 'timestamp' not in data_raw:
        #if timestamp not in dataframe then data come from fdit
        data_raw['timestamp'] = data_raw.apply(lambda row : pd.Timestamp(f"{row.date} {row.time}"), axis=1)
        
    t = Traffic(data_raw)
    
    if 'flight_id' in data_raw.columns:
        df_1 = next(iter(t)).data
        flight_id = df_1.flight_id.iloc[0]
        flight_names.append(flight_id)
        res_dataset = parser(df_1)
        
        for flight in t:
            
            df = flight.data
            if df.flight_id.iloc[0] not in (flight_id):
                flight_names.append(df.flight_id.iloc[0])
                ds = parser(df)
                res_dataset = res_dataset.concatenate(ds)
        return flight_names, res_dataset
    else:
        for flight in t:
            df = flight.data
            dataset = parser(df)
            return None, dataset
        
#%%

def get_thresholds(dataset, model):
    
    mean = []
    
    for tensor in dataset:
        data = tensor
        y_pred = model.predict(data)
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        test_mae_loss = np.mean(test_mae_loss, axis=-1)
        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        mean.append(test_score_df.loc[0].mean())
        
    mean = np.array(mean)    

    
    #Filtering out obvious outliers from training 

    # filtered_mean_asc = mean_asc[mean_asc < np.quantile(mean_asc, 0.98)]
    # filtered_mean_cru = mean_cru[mean_cru < np.quantile(mean_cru, 0.98)]
    # filtered_mean_des = mean_des[mean_des < np.quantile(mean_des, 0.98)]
                                 
    thresh = mean.mean() + 3*mean.std()
    
    # 0.09194569662213326
    return thresh
    
def get_metrics(flight_names, model, dataset) :
    flight_inc = 0
    results = pd.DataFrame(columns=['flight_name', 'accuracy', 'recall', 'fpr', 'f1'])
    for tensor in dataset:
        data, label = tensor
        y_pred = model.predict(data)
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        test_mae_loss = np.mean(test_mae_loss, axis=-1)
    
        test_score_df = pd.DataFrame()
        test_score_df['test_mae_loss'] = test_mae_loss
        test_score_df['label'] = label.numpy()
        
        test_score_df['anomaly'] = test_score_df['test_mae_loss'] > 0.09194569662213326
        test_score_df['anomaly'] = test_score_df['anomaly'].astype(int)
        
        r = tf.math.confusion_matrix(test_score_df['label'], test_score_df['anomaly'], num_classes=2)
        # m = tf.keras.metrics.Recall()
        # m.update_state(test_score_df['label'], test_score_df['anomaly'])
        # recall = m.result().numpy()
        
        # m = tf.keras.metrics.Precision()
        # m.update_state(test_score_df['label'], test_score_df['anomaly'])
        # precision = m.result().numpy()
        
        tp = r[1,1].numpy()
        fp = r[0,1].numpy()
        fn = r[1,0].numpy()
        tn = r[0,0].numpy()
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
  
        f1 = 2 * tp / (2*tp + fp + fn)

        res = pd.Series([flight_names[flight_inc], acc, recall, fpr, f1],  index=results.columns)
        results = results.append(res, ignore_index=True)
        flight_inc += 1
    #results['fpr'] = results.fpr.apply(lambda x : x.numpy())
    return results

#%%

def main():
    
    configParser = configparser.ConfigParser()
    #  configParser.optionxform makes sure we keep keys' case
    configParser.optionxform=str 
    configFilePath = r'../config/model_testing/model_testing.ini'
    configParser.read(configFilePath)
    
    training_path = configParser.get('options', 'train_path')
    scenarios = configParser._sections['scenario_folders']
    model_type = configParser.get('model', 'model_type')
    model_path = configParser.get('model', 'path')
    output_path = configParser.get('options', 'output_path')
    airport_refs = configParser._sections['airport_references']
    
    # Loading trained model from given path
    model = tf.keras.models.load_model(model_path)
    
    if model_type in ('autoencoder'):
        # Getting normalisation layers. Change depending on the model.
    
        for scenario_name, scenario_path in scenarios.items():
            
            output_root_folder = f'{output_path}{scenario_name}'
            if not os.path.exists(output_root_folder):
                os.mkdir(output_root_folder)
            
            for folder_path in glob.glob(f'{scenario_path}/*'):
                # Entering the loop to get results per flights
                
                print(f"processing {folder_path}")
                
                flight_name = os.path.basename(folder_path).split(".")[0]
                try :
                    dep_airp, arr_airp = eval(airport_refs[flight_name])
                except KeyError:
                    continue
                output_folder = f'{output_root_folder}/{flight_name}'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                
                results = pd.DataFrame(columns=['flight_name', 'accuracy', 'recall', 'fpr', 'f1'])
                for file in glob.glob(f"{folder_path}/*.pkl"):
                    filename = os.path.basename(file).split(".")[0]
                    data_raw = pd.read_pickle(file)
                    if scenario_name in ('hijack'):
                        data_raw['label'] = 0
                        data_raw.loc[data_raw.index > 3203, 'label'] = 1
                    flight_names, dataset = get_dataset(data_raw, dep_airp, arr_airp)
                    
                    if flight_names is None:
                        flight_names = [filename]
                    temp_results = get_metrics(flight_names, model, dataset)
                    results = results.append(temp_results)
                results.to_csv(f'{output_folder}/{flight_name}.csv', index=False)
                
if __name__ == "__main__":
   main()
   
   