#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:14:51 2020

@author: antoine chevrot
"""

import pandas as pd
from scifly.fdit import sbs_converter
import scifly
from traffic.core.traffic import Traffic, Flight
import matplotlib.pyplot as plt
import geopandas
import tensorflow as tf
import numpy as np
from pathlib import Path
import glob, os
import configparser

#%%

def flight_cleaning(t:Traffic) -> Traffic:
    
    def flight_id(gen):
        idx = 0
        for flight in gen:
            idx +=1
            yield Flight(scifly.data_cleaner(flight.assign_id(idx=idx).resample('2s').data))
    
    t.data.drop(['flight_id'], axis=1, inplace=True)
    return Traffic.from_flights(flight_id(t.iterate(by="60 minutes")))

def  data_plotting(t:Traffic, folder=None):
    world = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres'))
    for flight in t.iterate():
        data = flight.data
        
        with plt.style.context(("seaborn", "ggplot")):
            world.plot(figsize=(18,10),
                       color="white",
                       edgecolor = "blue");
        
            plt.scatter(data.longitude, data.latitude, s=15, color="red", alpha=0.3)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            if folder is not None:
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                plt.savefig(f"{folder}/{data.flight_id.iloc[0]}.png", format="png")
    
#%%

def bst_export(t:Traffic):
    inc = 1 # For naming the files
    for flight in t:
        sbs_data = sbs_converter(flight.data)
        sbs_data.to_csv(f'/Home/Users/achevrot/WORK/scifly/Examples/fdit/{inc}_{flight.data.icao24.iloc[0]}.bst', index=False, header=False)
        inc += 1    

# %%

def feature_extraction(t:Traffic) -> pd.DataFrame:
    
    # Next step is to transform into a tfrecord for TF model training
    # We first need to define and get all the features we want for our model :
    
    # In this example we will use :
    
    # altitude : original data (OD)
    # speed : OD
    # vertical_rate : OD
    # track_delta : difference between real track and optimal track
    # delta : distance in km between 2 consecutive messages
    # phase : the flight phase the message is in
    # label : legit or modified message (0 or 1)
    
    # Cleaning extra useless columns
    
    df = t.data
    df.drop(['lat_shift_c','lat_shift_f','lat_shift_c','onground', 'lon_shift_f', 'lon_shift_c'], axis=1, inplace=True)
    df.drop(["start","stop","track_unwrapped", "delta_close", "delta_far", "geoaltitude"], axis=1, inplace=True)
    
    # delta
    
    df['delta'] = df.groupby('flight_id').apply(scifly.distance_delta).values
    
    # phase
    df.reset_index(drop=True, inplace=True)
    phases = df.groupby("flight_id").apply(lambda flight : Flight(flight).phases(10).data.phase.replace('NA', None).replace('LEVEL', None))
    phases.index = phases.index.droplevel()
    df = pd.concat([df, phases], axis=1)
    
    # d_a, d_b, d_c, d_d
    
    
    from openap.phase import FlightPhase
    from vincenty import vincenty
    
    def distance_from_indice(df, indice_index):
        return df.apply(lambda x: vincenty(
        (x['latitude'],
         x['longitude']),
        (df.iloc[indice_index]['latitude'],
         df.iloc[indice_index]['longitude'])), axis = 1)
    
    def distance(df):
        fp = FlightPhase()
        fp.set_trajectory(
            (df.timestamp.values - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "s"),
            df.altitude.values,
            df.groundspeed.values,
            df.vertical_rate.values,
        )
        
        idx = fp.flight_phase_indices()
        d_a = distance_from_indice(df, 0)
        d_b = distance_from_indice(df, idx['CR']) if idx['CR'] is not None else None
        d_c = distance_from_indice(df, idx['DE']) if idx['DE'] is not None else None
        d_d = distance_from_indice(df, (idx['END']-1))
        return pd.concat([d_a, d_b, d_c, d_d], axis=1)
    
    
    distances = df.groupby("flight_id").apply(distance)
    df = pd.concat([df, distances], axis=1)
    df.rename(columns = {0:'d_a', 1:'d_b', 2:'d_c', 3:'d_d'},
                              inplace = True)
    # label
    
    df['label'] = 0
    
    # Final check to get rid of nan values that will disturb training
    
    df.dropna(inplace=True)
    return df
    
#%%

# # Serializing

# # Helper functions for the serialisation :

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(data: pd.DataFrame):

    """

    Creates a tf.Example message ready to be written to a file.

    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible

    # data type.

    flight = tf.train.Features(feature={
      'icao': _bytes_feature(data.icao24.astype(str).iloc[0].encode()),
      'cal': _bytes_feature(data.callsign.astype(str).iloc[0].encode())
    })

    alt_features = data.altitude.apply(_float_feature).values
    label_features = data.label.apply(_float_feature).values
    lat_features = data.latitude.apply(_float_feature).values
    lon_features = data.longitude.apply(_float_feature).values
    spd_features = data.groundspeed.apply(_float_feature).values
    delta_features = data.delta.apply(_float_feature).values
    vert_rate_features = data.vertical_rate.apply(_float_feature).values
    d_a_features = data.d_a.apply(_float_feature).values
    d_b_features = data.d_b.apply(_float_feature).values
    d_c_features = data.d_c.apply(_float_feature).values
    d_d_features = data.d_d.apply(_float_feature).values
    hdg_features = data.track.apply(_float_feature).values
    
    
    alt = tf.train.FeatureList(feature=alt_features)
    lat = tf.train.FeatureList(feature=lat_features)
    lon = tf.train.FeatureList(feature=lon_features)
    vert_rate = tf.train.FeatureList(feature=vert_rate_features)
    hdg = tf.train.FeatureList(feature=hdg_features)
    spd = tf.train.FeatureList(feature=spd_features)
    delta = tf.train.FeatureList(feature=delta_features)
    d_a = tf.train.FeatureList(feature=d_a_features)
    d_b = tf.train.FeatureList(feature=d_b_features)
    d_c = tf.train.FeatureList(feature=d_c_features)
    d_d = tf.train.FeatureList(feature=d_d_features)
    label = tf.train.FeatureList(feature=label_features)

    messages = tf.train.FeatureLists(feature_list={
        'alt': alt,
        'label': label,
        'lat': lat,
        'lon': lon,
        'vert_rate': vert_rate,
        'hdg': hdg,
        'spd': spd,
        'delta': delta,
        'd_a': d_a,
        'd_b': d_b,
        'd_c': d_c,
        'd_d': d_d,
      })

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.SequenceExample(context=flight,
                                              feature_lists=messages)
    return example_proto.SerializeToString()
    
# %%

# For an import from FDI-T :
def from_fdit():
    cols = [
        "msg",
        "msg_type",
        "msg_type2",
        "icao24_dec",
        "icao24",
        "icao24_2",
        "date",
        "time",
        "date_2",
        "time_2",
        "callsign",
        "altitude",
        "groundspeed",
        "track",
        "latitude",
        "longitude",
        "vertical_rate",
        "squawk",
        "alert",
        "emergency",
        "spi",
        "surface",
        "label",
        "ano_type"
    ]
    
    # # For an import from FDI-T :
        
    # p = Path("/mnt/meso/Examples/MultiFlight/Test_data/Weather_Avoidance/")
    p = Path("/mnt/meso/scifly/Examples/MultiFlight/Test_data/Weather_Avoidance/")
    
    for file in p.glob('anomalous/*.bst'):
    
    
        sbs_data_ano = pd.read_csv(file, names=cols)
        sbs_data_legit = pd.read_csv(p / 'legit' / file.name, names=cols)
        # Track delta
        
        track_delta = scifly.track_delta(sbs_data_legit)
        sbs_data_ano['track_delta'] = track_delta[:len(sbs_data_ano)]
        
        # delta
        
        sbs_data_ano['delta'] = scifly.distance_delta(sbs_data_ano)
        
        # phase
        
        # Retrouver les timestamps
        sbs_data_legit['timestamp'] = sbs_data_legit.apply(lambda row : pd.Timestamp(f"{row.date} {row.time}"), axis=1)
        sbs_data_ano['phase'] = Flight(sbs_data_legit).phases().data.phase.values[:len(sbs_data_ano)]
        sbs_data_ano['phase'] = sbs_data_ano.phase.replace('NA', None) #replace NA and LEVEL by None and automatically pad the serie
        sbs_data_ano['phase'] = sbs_data_ano.phase.replace('LEVEL', None)
        
        # Final check to get rid of nan values that will disturb training
        sbs_data_ano.drop(["squawk"], axis=1, inplace=True)
        sbs_data_ano.dropna(inplace=True)
        
        serialized_data = serialize_example(sbs_data_ano)
    
        # # Writing into tfrecord files
    
        with tf.io.TFRecordWriter((p / f'tfrecord/{file.stem}.tfrecord').__str__()) as writer:
            writer.write(serialized_data)

#%%

def main():
    
    configParser = configparser.RawConfigParser()  
    configFilePath = r'./config/data_cleaning.ini'
    configParser.read(configFilePath)

    raw_data_folder = configParser.get('paths', 'raw_data_path')
    tfrecord_folder = configParser.get('paths', 'tfrecord_path')
    for file in glob.glob(raw_data_folder):
        
        df_brut = Traffic.from_file(file)
        cleaned_traffic = flight_cleaning(df_brut)
        
        # data_plotting(cleaned_traffic, f'{tfrecord_folder}{os.path.basename(file).split(".")[0]}')
        df = feature_extraction(cleaned_traffic)
        
        # serialized_data = df.groupby('flight_id').apply(serialize_example)if idx['CR'] is not None else None
        serialized_data = serialize_example(df)

        # Writing into tfrecord files

        
        with tf.io.TFRecordWriter(f'{tfrecord_folder}{os.path.basename(file).split(".")[0]}.tfrecord') as writer:
            writer.write(serialized_data)
        
        

if __name__ == "__main__":
   main()