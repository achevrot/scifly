#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:17:15 2021

@author: antoine
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import pickle
import configparser
from traffic.core.traffic import Traffic, Flight
from SVDD.src.svdd import SVDD
from SVDD.src.visualize import Visualization as draw
from vincenty import vincenty

# import seaborn
# seaborn.set(style='ticks')
# %%

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
        'label': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
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
        (departure_coordinate, arrival_coordinate),  # pass these args to the above function.
        tf.float32)  # the return type is `tf.string`.
    return tf.reshape(flight_length, ())  # The result is a scalar


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
        tf.data.TFRecordDataset, cycle_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda x: (tensor_windowing(x)))
    dataset = dataset.unbatch().batch(
        params['BATCH_SIZE'], drop_remainder=True).prefetch(1)
    dataset = dataset.map(lambda x: (
        tf.ensure_shape(x,
                        [None, params["TIME_STEPS"],
                         params["FEATURE_NUMBERS"]])))
    return dataset
# %%


def normalize(input_df, departure_coordinate, arrival_coordinate):

    df = input_df.copy()
    flight_length = vincenty(departure_coordinate, arrival_coordinate)
    df['altitude'] = df['altitude'] / 40000

    # lat_min = min(departure_coordinate[0], arrival_coordinate[0])
    # lat_max = max(departure_coordinate[0], arrival_coordinate[0])
    # lon_min = min(departure_coordinate[1], arrival_coordinate[1])
    # lon_max = max(departure_coordinate[1], arrival_coordinate[1])
    lat_min = tf.reduce_min(df['latitude'])
    lat_max = tf.reduce_max(df['latitude'])
    lon_min = tf.reduce_min(df['longitude'])
    lon_max = tf.reduce_max(df['longitude'])

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

# %%


def get_dataset(data_raw, departure_coordinate, arrival_coordinate):

    flight_names = []

    def parser(df):
        label = df.pop('label')
        df = normalize(df, departure_coordinate, arrival_coordinate)
        df = df[['altitude', 'latitude', 'longitude', 'groundspeed']]
        df.reset_index(drop=True, inplace=True)

        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        dataset = dataset.window(params["TIME_STEPS"],
                                 shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x, y: (
            tf.data.Dataset.zip((x.batch(params["TIME_STEPS"],
                                         drop_remainder=True),
                                 y.batch(params["TIME_STEPS"],
                                         drop_remainder=True)))))
        dataset = dataset.map(lambda x, y: (
            tf.reshape(x,
                       [params["TIME_STEPS"],
                        params["FEATURE_NUMBERS"]]),
            tf.reshape(tf.reduce_max(y, axis=0), [-1]))).batch(len(df))
        return dataset

    if 'timestamp' not in data_raw:
        # if timestamp not in dataframe then data come from fdit
        data_raw['timestamp'] = data_raw.apply(lambda row: pd.Timestamp(
            f"{row.date} {row.time}"), axis=1)

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
# %%


def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def train_svdds(model, dataset):

    df = pd.DataFrame()
    for tensor in dataset.take(100):
        data = tensor
        # t = len(data)//params["BATCH_SIZE"]*params["BATCH_SIZE"]
        # data = data[t:]
        y_pred = model.predict(data, batch_size=params["BATCH_SIZE"])

        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        # test_mae_loss = np.mean(test_mae_loss, axis=-1)

        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        test_score_df['label'] = 1

        df = df.append(test_score_df, ignore_index=True)

    # set SVDD parameters
    parameters = {"positive penalty": 1,
                  "negative penalty": [],
                  "kernel": {"type": 'gauss', "width": 1/4},
                  "option": {"display": 'on'}}

    svdd = SVDD(parameters)
    df = reject_outliers(df)
    df = df.sample(10000)

    trainData = df[[0, 1, 2, 3]].reset_index(drop=True)
    # trainLabel = df['label'].values.reshape((-1, 1))
    svdd.train(trainData.values, np.array([1]*len(trainData),
                                          dtype=np.uint8).reshape((-1, 1)))

    with open('svdd.pickle', 'wb') as file:
        pickle.dump(svdd, file)


# %%


def reject_outliers(input_df):
    q = input_df[0].quantile(0.99)
    df = input_df[input_df[0] < q]
    q = df[1].quantile(0.99)
    df = df[df[1] < q]
    q = df[2].quantile(0.99)
    df = df[df[2] < q]
    q = df[3].quantile(0.99)
    df = df[df[3] < q]
    return df
# %%


def get_metrics_svdd(flight_names, model, dataset, svdd):

    thresh = svdd.model['radius']

    flight_inc = 0
    results = pd.DataFrame(columns=['flight_name', 'accuracy', 'recall', 'fpr', 'f1'])
    for tensor in dataset:
        data, label = tensor
        t = len(data)//params["BATCH_SIZE"]*params["BATCH_SIZE"]
        data = data[:t]
        label = label[:t]
        if t == 0:
            continue
        y_pred = model.predict(data, batch_size=256)
        test_mae_loss = np.mean(np.abs(y_pred - data), axis=1)
        # test_mae_loss = np.mean(test_mae_loss, axis=-1)

        test_score_df = pd.DataFrame()
        test_score_df = pd.concat([test_score_df, pd.DataFrame(test_mae_loss)], axis=1)
        test_score_df['label'] = label.numpy()

        svdd_data = test_score_df.copy(deep=True)
        svdd_data['label'].replace(1, -1, inplace=True)
        svdd_data['label'].replace(0, 1, inplace=True)

        testData = svdd_data[[0, 1, 2, 3]].reset_index(drop=True)
        testLabel = svdd_data['label'].values.reshape((-1, 1))
        distance, accuracy = svdd.test(testData.values, testLabel)
        # draw.testResult(svdd, distance)

        test_score_df['anomaly'] = np.reshape(1*(distance > thresh), (-1, 1))
        test_score_df['distance'] = np.reshape(distance, (-1, 1))

        acc = 1 - np.abs(test_score_df['label'] -
                         test_score_df['anomaly']).sum() / len(test_score_df)

        r = tf.math.confusion_matrix(test_score_df['label'],
                                     test_score_df['anomaly'], num_classes=2)

        tp = r[1, 1].numpy()
        fp = r[0, 1].numpy()
        fn = r[1, 0].numpy()
        tn = r[0, 0].numpy()

        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)

        f1 = 2 * tp / (2*tp + fp + fn)

        res = pd.Series([flight_names[flight_inc], acc, recall, fpr, f1], index=results.columns)
        results = results.append(res, ignore_index=True)
        flight_inc += 1
    return results

# %%


def main():

    configParser = configparser.ConfigParser()
    #  configParser.optionxform makes sure we keep keys' case
    configParser.optionxform = str
    configFilePath = r'../config/model_testing/model_testing.ini'
    configParser.read(configFilePath)

    # training_path = configParser.get('options', 'train_path')
    scenarios = configParser._sections['scenario_folders']
    model_type = configParser.get('model', 'model_type')
    model_path = configParser.get('model', 'path')
    output_path = configParser.get('options', 'output_path')
    svdd_option = configParser.get('options', 'svdd')
    airport_refs = configParser._sections['airport_references']

    # Loading trained model from given path
    model = tf.keras.models.load_model(model_path, compile=False)

    if eval(svdd_option):
        dataset = data_preparation("../../../../Data/Training_data/autoencoder/*tfrecord")
        train_svdds(model, dataset)

    if model_type in ('autoencoder'):
        # Getting normalisation layers. Change depending on the model.
        with open("/mnt/meso/scifly/Models/vae/scripts_SGE/svdd.pickle", 'rb') as file:
            svdd = pickle.load(file)

        for scenario_name, scenario_path in scenarios.items():

            output_root_folder = f'{output_path}{scenario_name}'
            if not os.path.exists(output_root_folder):
                os.mkdir(output_root_folder)

            for folder_path in glob.glob(f'{scenario_path}/*'):
                # Entering the loop to get results per flights

                print(f"processing {folder_path}")

                flight_name = os.path.basename(folder_path).split(".")[0]
                try:
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
                    # data_raw['label'] = 0
                    # data_raw.loc[data_raw.index > 3203, 'label'] = 1
                    flight_names, dataset = get_dataset(data_raw, dep_airp, arr_airp)

                    if flight_names is None:
                        flight_names = [filename]
                    temp_results = get_metrics_svdd(flight_names, model, dataset, svdd)
                    results = results.append(temp_results)
                results.to_csv(f'{output_folder}/{flight_name}.csv', index=False)


if __name__ == "__main__":
    main()
