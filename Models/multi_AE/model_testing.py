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
import pickle
import configparser
from traffic.core.traffic import Traffic, Flight
from SVDD.src.svdd import SVDD
from SVDD.src.visualize import Visualization as draw

# import seaborn
# seaborn.set(style='ticks')
#%%

params = {}
params["BATCH_SIZE"] = 512
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

#%%

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
                        sizes=[1, 30, 1, 1],
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
    ds = result_ds.unbatch().shuffle(params['BATCH_SIZE']*10).batch(params['BATCH_SIZE'], drop_remainder=True).prefetch(1)
    ds = ds.map(lambda x, y: (tf.ensure_shape(x,[None,params["TIME_STEPS"],params["FEATURE_NUMBERS"]]),tf.ensure_shape(tf.cast(tf.reshape(tf.reduce_max(y, axis=1),[-1]),tf.float32),[params['BATCH_SIZE'],])))
    return ds


#%%

def get_dataset(data_raw, phase_norm_layer, feature_norm_layer):
    
    flight_names=[]
    
    def parser(df_input):
        df = df_input.copy(deep=True)
        phase = df.pop('phase')
        label = df.pop('label')
        df = df[['altitude', 'delta', 'groundspeed', 'track_delta', 'vertical_rate']]
        df.reset_index(drop=True, inplace=True)
        
        dataset = tf.data.Dataset.from_tensor_slices((df.values, phase.values, label.values))
        dataset = dataset.window(30, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x,y,z : tf.data.Dataset.zip((x.batch(30, drop_remainder=True), y.batch(30, drop_remainder=True), z.batch(30, drop_remainder=True))))
        dataset = dataset.map(lambda x,y,z : (feature_norm_layer(x), phase_norm_layer(y), z))
        dataset = dataset.map(lambda x, y, z: (tf.reshape(x,[30, 5]), tf.reshape(tf.reduce_max(y, axis=0),[-1]), tf.reshape(tf.reduce_max(z, axis=0),[-1]))).batch(len(df))
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
    
    mean_asc = []
    mean_cru = []
    mean_des = []
    
    for tensor in dataset:
        data, phase = tensor
        y_pred = model.predict((data, phase))
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        test_mae_loss = np.mean(test_mae_loss, axis=-1)
        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        test_score_df['phase'] = phase.numpy()
        
        mean_asc.append(test_score_df.loc[test_score_df['phase'] == 1, 0].mean())
        mean_cru.append(test_score_df.loc[test_score_df['phase'] == 2, 0].mean())
        mean_des.append(test_score_df.loc[test_score_df['phase'] == 3, 0].mean())
        
    mean_asc = np.array(mean_asc)    
    mean_cru = np.array(mean_cru)    
    mean_des = np.array(mean_des)    
    
    mean_asc = mean_asc[~np.isnan(mean_asc)]
    mean_cru = mean_cru[~np.isnan(mean_cru)]
    mean_des = mean_des[~np.isnan(mean_des)]
    
    #Filtering out obvious outliers from training 

    # filtered_mean_asc = mean_asc[mean_asc < np.quantile(mean_asc, 0.98)]
    # filtered_mean_cru = mean_cru[mean_cru < np.quantile(mean_cru, 0.98)]
    # filtered_mean_des = mean_des[mean_des < np.quantile(mean_des, 0.98)]
                                 
    asc_thresh = mean_asc.mean() + 3*mean_asc.std()
    cru_thresh = mean_asc.mean() + 3*mean_asc.std()
    des_thresh = mean_asc.mean() + 3*mean_asc.std()
    
    asc_thresh = mean_asc.mean() + 3*mean_asc.std()
    cru_thresh = mean_cru.mean() + 3*mean_cru.std()
    des_thresh = mean_des.mean() + 3*mean_des.std()
    
    
    # 0.10656501600294294, 0.07988857754414519, 0.11254387042295558
    return asc_thresh, cru_thresh, des_thresh
    

def get_metrics(flight_names, model, dataset) :
    flight_inc = 0
    results = pd.DataFrame(columns=['flight_name', 'accuracy', 'recall', 'fpr', 'f1'])
    for tensor in dataset:
        data, phase, label = tensor
        y_pred = model.predict((data, phase))
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        test_mae_loss = np.mean(test_mae_loss, axis=-1)
    
        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        test_score_df['label'] = label.numpy()
        test_score_df['phase'] = phase.numpy()
        
        test_score_df['anomaly'] = np.zeros(len(test_score_df))
        test_score_df.loc[test_score_df['phase'] == 1, 'anomaly'] = test_score_df.loc[test_score_df['phase'] == 1, 0] > 0.10656501600294294
        test_score_df.loc[test_score_df['phase'] == 2, 'anomaly'] = test_score_df.loc[test_score_df['phase'] == 2, 0] > 0.07988857754414519
        test_score_df.loc[test_score_df['phase'] == 3, 'anomaly'] = test_score_df.loc[test_score_df['phase'] == 3, 0] > 0.11254387042295558
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
    # results['fpr'] = results.fpr.apply(lambda x : x.numpy())
    return results

def train_svdds(model, dataset):
        
    ascending_df = pd.DataFrame(columns=['label', 'phase'])
    cruising_df = pd.DataFrame(columns=['label', 'phase'])
    descending_df = pd.DataFrame(columns=['label', 'phase'])
    
    for tensor in dataset.take(1000):
        data, phase = tensor
        y_pred = model.predict((data, phase))
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        # test_mae_loss = np.mean(test_mae_loss, axis=-1)
    
        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        test_score_df['label'] = 1
        test_score_df['phase'] = phase.numpy()
        
        ascending_df = ascending_df.append(test_score_df.loc[test_score_df['phase'] == 1], ignore_index=True)
        cruising_df = cruising_df.append(test_score_df.loc[test_score_df['phase'] == 2], ignore_index=True)
        descending_df = descending_df.append(test_score_df.loc[test_score_df['phase'] == 3], ignore_index=True)
        
    # set SVDD parameters
    parameters = {"positive penalty": 1,
              "negative penalty": [],
              "kernel": {"type": 'gauss', "width": 1/5},
              "option": {"display": 'on'}}
    
    svdd_asc = SVDD(parameters)
    svdd_cru = SVDD(parameters)
    svdd_des = SVDD(parameters)
    
    ascending_df = reject_outliers(ascending_df)
    cruising_df = reject_outliers(cruising_df)
    descending_df = reject_outliers(descending_df)
      
    ascending_df = ascending_df.sample(10000)
    cruising_df = cruising_df.sample(10000)
    descending_df = descending_df.sample(10000)
    
    trainData_asc, trainLabel_asc = ascending_df[[0,1,2,3,4]].reset_index(drop=True), ascending_df['label'].values.reshape((-1,1))
    trainData_crui, trainLabel_crui = cruising_df[[0,1,2,3,4]], cruising_df['label'].values.reshape((-1,1))
    trainData_desc, trainLabel_desc = descending_df[[0,1,2,3,4]], descending_df['label'].values.reshape((-1,1))
    
    svdd_asc.train(trainData_asc.values, np.array([1]*len(trainData_asc), dtype=np.uint8).reshape((-1,1)))
    svdd_cru.train(trainData_crui.values, np.array([1]*len(trainData_crui), dtype=np.uint8).reshape((-1,1)))
    svdd_des.train(trainData_desc.values, np.array([1]*len(trainData_desc), dtype=np.uint8).reshape((-1,1)))


    
    with open(f'svdd_asc.pickle', 'wb') as file:
        pickle.dump(svdd_asc, file)
    with open(f'svdd_crui.pickle', 'wb') as file:
        pickle.dump(svdd_cru, file)
    with open(f'svdd_desc.pickle', 'wb') as file:
        pickle.dump(svdd_des, file)


#%%

def reject_outliers(input_df):
    q = input_df[0].quantile(0.99)
    df = input_df[input_df[0] < q]
    q = df[1].quantile(0.99)
    df = df[df[1] < q]
    q = df[2].quantile(0.99)
    df = df[df[2] < q]
    q = df[3].quantile(0.99)
    df = df[df[3] < q]
    q = df[4].quantile(0.99)
    df = df[df[4] < q]
    return df

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
    
    # Loading trained model from given path
    model = tf.keras.models.load_model(model_path, compile=False)
    
    if model_type in ('multi_ae'):
        # Getting normalisation layers. Change depending on the model.
        phase_norm_layer, feature_norm_layer = get_norm_layer(f"{training_path}")
    
        for scenario_name, scenario_path in scenarios.items():
            
            output_root_folder = f'{output_path}{scenario_name}'
            if not os.path.exists(output_root_folder):
                os.mkdir(output_root_folder)
            
            for folder_path in glob.glob(f'{scenario_path}/*'):
                # Entering the loop to get results per flights
                
                print(f"processing {folder_path}")
                
                flight_name = os.path.basename(folder_path).split(".")[0]
                output_folder = f'{output_root_folder}/{flight_name}'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                
                results = pd.DataFrame(columns=['flight_name', 'accuracy', 'recall', 'fpr', 'f1'])
                for file in glob.glob(f"{folder_path}/**/*.pkl", recursive=True):
                    filename = os.path.basename(file).split(".")[0]
                    data_raw = pd.read_pickle(file)
                    if scenario_name in ('hijack'):
                        data_raw.loc[data_raw.index > 3205, 'label'] = 1
                    flight_names, dataset = get_dataset(data_raw, phase_norm_layer, feature_norm_layer)
                    
                    if flight_names is None:
                        flight_names = [filename]
                    temp_results = get_metrics(flight_names, model, dataset)
                    results = results.append(temp_results)
                results.to_csv(f'{output_folder}/{flight_name}.csv', index=False)

if __name__ == "__main__":
   main()

