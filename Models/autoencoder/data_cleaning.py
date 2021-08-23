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
import glob, os
import configparser
from openap.phase import FlightPhase
from vincenty import vincenty

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

def feature_extraction(t:Traffic, airport_refs) -> pd.DataFrame:
    
    # Next step is to transform into a tfrecord for TF model training
    # We first need to define and get all the features we want for our model :
    
    # In this example we will use :
    
    # altitude : original data (OD)
    # speed : OD
    # vertical_rate : OD
    # track_delta : difference between real track and optimal track
    # delta : distance in km between 2 consecutive messages
    # label : legit or modified message (0 or 1)
    
    # Cleaning extra useless columns
    
    df = t.data
    # df.drop(['lat_shift_c','lat_shift_f','lat_shift_c','onground', 'lon_shift_f', 'lon_shift_c'], axis=1, inplace=True)
    # df.drop(["start","stop","track_unwrapped", "delta_close", "delta_far", "geoaltitude"], axis=1, inplace=True)
    
    # delta
    
    df['delta'] = scifly.distance_delta(df).values
    
    # d_a, d_b, d_c, d_d
    
    dep_airp, arr_airp = airport_refs
    
    def distance_from_indice(df, indice_index):
        return df.apply(lambda x: vincenty(
        (x['latitude'],
         x['longitude']),
        (df.iloc[indice_index]['latitude'],
         df.iloc[indice_index]['longitude'])), axis = 1)
    
    
    
    def distance(df):
        
        df.reset_index(drop=True, inplace=True)
        b_p, c_p  = get_phase_points(df)
        
        # Then check if they are not too far from airports based on history
        b_hist = get_dist_index(df, dep_airp, 300)
        c_hist = get_dist_index(df, arr_airp, 300, side='right')
        
        # In case the flight is less than 300 km.
        if b_hist is None or c_hist is None:
            b, c =  b_p, c_p
        else:
            b = min((b_p,b_hist))
            c = max((c_p, c_hist))
        
        d_b = distance_from_indice(df, b)
        d_c = distance_from_indice(df, c)
        
        results = pd.concat([df, d_b, d_c], axis=1)
        return results
    
    df['d_a'] = df.apply(lambda x: vincenty(
        (x['latitude'], x['longitude']), dep_airp), axis = 1)
    
    df['d_d'] = df.apply(lambda x: vincenty(
        (x['latitude'], x['longitude']), arr_airp), axis = 1)
    
    
    # data = df.groupby("flight_id").apply(distance)
    data = distance(df)
    data.rename(columns = {0:'d_b', 1:'d_c'}, inplace = True)
    
    if data['d_a'].min() > 300:
        data['d_b'] = data['d_a'] - 300
        
    if data['d_d'].min() > 300:
        data['d_c'] = data['d_d'] - 300
   
    # label
    
    data['label'] = 0
    
    # Final check to get rid of nan values that will disturb training
    
    data.dropna(inplace=True)
    return data
    
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

def serialize_example(data: pd.DataFrame, airport_refs):

    """

    Creates a tf.Example message ready to be written to a file.

    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible

    # data type.

    flight = tf.train.Features(feature={
      'icao': _bytes_feature(data.icao24.astype(str).iloc[0].encode()),
      'cal': _bytes_feature(data.callsign.astype(str).iloc[0].encode()),
      'dep_airport_lat': _float_feature(airport_refs[0][0]),
      'dep_airport_lon': _float_feature(airport_refs[0][1]),
      'arr_airport_lat': _float_feature(airport_refs[1][0]),
      'arr_airport_lon': _float_feature(airport_refs[1][1]),
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

def get_phase_points(df):

    fp = FlightPhase()
    
    fp.set_trajectory(
    (df.timestamp.values - np.datetime64("1970-01-01"))
    / np.timedelta64(1, "s"),
    df.altitude.values,
    df.groundspeed.values,
    df.vertical_rate.values,
    )
    labels = np.array(fp.phaselabel())
    
    idx_cl = np.nonzero(np.array(labels) == 'CL')[0]
    idx_cr = np.nonzero(np.array(labels) == 'CR')[0]
    
    b = 0 if idx_cl.size == 0 else idx_cl[-1]
    c = np.array(labels).size - 1 if idx_cr.size == 0 else idx_cr[-1]
    return b, c

def get_dist_index(df_input, ref, ref_dist, side='left'):
    
    df = df_input.copy()
    
    if side == 'right':
        df = df.iloc[::-1]
    
    for i, row in df.iterrows():
        dist = vincenty((row.latitude, row.longitude), ref)
        if dist > ref_dist:
            return i-1

# For an import from FDI-T :
def from_bst(raw_data_folder, airport_refs, output_folder):
    
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
    
    for folder_name, refs in airport_refs.items():
        
        # folder_name = 'omaha_phoenix'
        # refs = '[(41.30532080580088, -95.89515895221132),(33.43109089632534, -112.01082600841279)]'
        # output_folder = "/mnt/meso/Data/Testing_data/Crash/multi_ae/test/"
        # raw_data_folder = "/mnt/meso/Data/Testing_data/Crash/FDIT/"
        # ref_folder = "/mnt/meso/Data/Testing_data/Legit_data_2021/multi_ae/"
        
        dep_airp, arr_airp = eval(refs) 
        inc = 0
        
        if not os.path.exists(f'{output_folder}{folder_name}'):
            os.mkdir(f'{output_folder}{folder_name}')
        for file in glob.glob(f"{raw_data_folder}{folder_name}/**/*.bst", recursive=True):
            print(file)
            df = pd.read_csv(file, names=cols, header=0)
            
            # Track delta
            track_delta = scifly.track_delta(df, arr_airp)
            df['track_delta'] = list(track_delta)
            
            # delta
            df['delta'] = scifly.distance_delta(df).values
            
            # d_a, d_b, d_c, d_d
            
            df['timestamp'] = df.apply(lambda row : pd.Timestamp(f"{row.date} {row.time}"), axis=1)
            
            def distance_from_indice(df, indice_index):
                return df.apply(lambda x: vincenty(
                (x['latitude'],
                 x['longitude']),
                (df.iloc[indice_index]['latitude'],
                 df.iloc[indice_index]['longitude'])), axis = 1)
            
            
            
            def distance(df):
                b_p, c_p  = get_phase_points(df)
                
                # Then check if they are not too far from airports based on history
                b_hist = get_dist_index(df, dep_airp, 300)
                c_hist = get_dist_index(df, arr_airp, 300, side='right')
                
                # In case the flight is less than 300 km.
                if b_hist is None or c_hist is None:
                    b, c =  b_p, c_p
                else:
                    b = min((b_p,b_hist))
                    c = max((c_p, c_hist))
                
                d_b = distance_from_indice(df, b)
                d_c = distance_from_indice(df, c)
                
                distance = pd.concat([d_b, d_c], axis=1)
                distance.rename(columns = {0:'d_b', 1:'d_c'}, inplace = True)
                return distance
            
            df['d_a'] = df.apply(lambda x: vincenty(
                (x['latitude'], x['longitude']), dep_airp), axis = 1)
            
            df['d_d'] = df.apply(lambda x: vincenty(
                (x['latitude'], x['longitude']), arr_airp), axis = 1)
            
            distance = distance(df)
            
            if df['d_a'].min() > 300:
                df['d_b'] = df['d_a'] - 300
            else:
                df['d_b'] = distance['d_b']
                
            if df['d_d'].min() > 300:
                df['d_c'] = df['d_d'] - 300
            else:
                df['d_c'] = distance['d_c']
                
            df.drop(['squawk'], axis=1, inplace=True)
            df.dropna(inplace=True)
            if df.empty:
                continue
            df.to_pickle(f'{output_folder}{folder_name}/{inc}_{df.icao24.iloc[0]}.pkl')
            inc += 1
#%%

def main():
    
    configParser = configparser.ConfigParser()
    #  configParser.optionxform makes sure we keep keys' case
    configParser.optionxform=str 
    configFilePath = r'../config/data_cleaning.ini'
    configParser.read(configFilePath)
    raw_data_folder = configParser.get('paths', 'raw_data_path')
    airport_refs = configParser._sections['airport_references']
    input_type = configParser.get('options', 'input_type')
    output_type = configParser.get('options', 'output_type')
    output_folder = configParser.get('paths', 'output_folder')
    
    if input_type in ('bst', 'fdit'):
        from_bst(raw_data_folder, airport_refs, output_folder)
    else:
        for file in glob.glob(raw_data_folder):
            basename = os.path.basename(file).split(".")[0]
            
            try:
                eval(airport_refs[basename])
            except KeyError:
                print(f'No references for {basename}. Skipping ...')
                continue
                
            print(f'cleaning {file} ...')
            
            if output_type in ('tfrecord'):
                if os.path.exists(f'{output_folder}{basename}.tfrecord'):
                    continue
                else:
                    with tf.io.TFRecordWriter(f'{output_folder}{basename}.tfrecord') as writer:
                        print(f'processing {basename} ...')
                        df_brut = Traffic.from_file(file)
                        cleaned_traffic = flight_cleaning(df_brut)
                        for flight in cleaned_traffic:
                            
                            df = flight.data.copy(deep=True)
                            df = feature_extraction(flight, eval(airport_refs[basename]))
                            if not df.empty:
                                serialized_data = serialize_example(df, eval(airport_refs[basename]))
                                writer.write(serialized_data)
                            
            elif output_type in ('pickle', 'pkl'):
                
                df_brut = Traffic.from_file(file)
                cleaned_traffic = flight_cleaning(df_brut)
                df = cleaned_traffic.data.copy(deep=True)
                df = feature_extraction(cleaned_traffic, eval(airport_refs[basename]))
                df.to_pickle(f'{output_folder}{basename}.pkl')
                
            elif output_type in ('bst', 'fdit'):
                df_brut = Traffic.from_file(file)
                cleaned_traffic = flight_cleaning(df_brut)
                bst_export(cleaned_traffic, basename, output_folder)
            
        

if __name__ == "__main__":
   main()
