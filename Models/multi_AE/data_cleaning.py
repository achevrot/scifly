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
from vincenty import vincenty
from openap.phase import FlightPhase
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

def bst_export(t:Traffic, basename, output_folder):
    inc = 1 # For naming the files
    if not os.path.exists(f'{output_folder}{basename}'):
        os.mkdir(f'{output_folder}{basename}')
    for flight in t:
        sbs_data = sbs_converter(flight.data)
        sbs_data.to_csv(f'{output_folder}{basename}/{inc}_{flight.data.icao24.iloc[0]}.bst', index=False, header=False)
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
    # phase : the flight phase the message is in
    # label : legit or modified message (0 or 1)
    
    # Cleaning extra useless columns
    
    df = t.data
    df.drop(['lat_shift_c','lat_shift_f','lat_shift_c','onground', 'lon_shift_f', 'lon_shift_c'], axis=1, inplace=True)
    df.drop(["start","stop","track_unwrapped", "delta_close", "delta_far", "geoaltitude"], axis=1, inplace=True)
    
    # Track delta
    
    dep_airp, arr_airp = airport_refs
    
    # track_delta = df.groupby('flight_id').apply(scifly.track_delta, arr_airp)
    track_delta = scifly.track_delta(df, arr_airp)
    df.sort_values(['timestamp'], inplace=True)
    df['track_delta'] = track_delta
    
    # delta
    
    # df['delta'] = df.groupby('flight_id').apply(scifly.distance_delta).values
    df['delta'] = scifly.distance_delta(df).values
    
    # phase
    
    df.reset_index(drop=True, inplace=True)
    # phases = df.groupby("flight_id").apply(lambda flight : Flight(flight).phases(10).data.phase.replace('NA', None).replace('LEVEL', None))
    phases = Flight(df).phases(10).data.phase.replace('NA', None).replace('LEVEL', None)
    # phases.index = phases.index.droplevel()
    df = pd.concat([df, phases], axis=1)
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
    spd_features = data.groundspeed.apply(_float_feature).values
    lat_features = data.latitude.apply(_float_feature).values
    lon_features = data.longitude.apply(_float_feature).values
    delta_features = data.delta.apply(_float_feature).values
    track_delta_features = data.track_delta.apply(_float_feature).values
    vert_rate_features = data.vertical_rate.apply(_float_feature).values
    phase_features = data.phase.str.encode("utf-8").apply(_bytes_feature).values
    label_features = data.label.apply(_float_feature).values
    
    alt = tf.train.FeatureList(feature=alt_features)
    lat = tf.train.FeatureList(feature=lat_features)
    lon = tf.train.FeatureList(feature=lon_features)
    spd = tf.train.FeatureList(feature=spd_features)
    delta = tf.train.FeatureList(feature=delta_features)
    track_delta = tf.train.FeatureList(feature=track_delta_features)
    vert_rate = tf.train.FeatureList(feature=vert_rate_features)
    phase = tf.train.FeatureList(feature=phase_features)
    label = tf.train.FeatureList(feature=label_features)

    messages = tf.train.FeatureLists(feature_list={
        'alt': alt,
        'delta': delta,
        'lat': lat,
        'lon': lon,
        'label': label,
        'phase': phase,
        'track_delta': track_delta,
        'spd': spd,
        'vert_rate': vert_rate,
      
      })

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.SequenceExample(context=flight,
                                              feature_lists=messages)
    return example_proto.SerializeToString()
    
# %%

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
        
        # hist_phase = phase_dist(f"{ref_folder}{folder_name}.pkl", dep_airp, arr_airp)
        
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
            
            # phase
            # Get the phase from Junzi algorithm            
            df['timestamp'] = df.apply(lambda row : pd.Timestamp(f"{row.date} {row.time}"), axis=1)
            b_p, c_p = get_phase_points(df)
            
            # Then check if they are not too far from airports based on history
            b_hist = get_dist_index(df, dep_airp, 300)
            c_hist = get_dist_index(df, arr_airp, 300, side='right')
            
            # In case the flight is less than 300 km.
            if b_hist is None or c_hist is None:
                b, c =  b_p, c_p
            else:
                b = min((b_p,b_hist))
                c = max((c_p, c_hist))
                
            df.loc[df.index.intersection(range(int(b))), 'phase'] = "CLIMB"
            df.loc[df.index.intersection(range(int(b), int(c))), 'phase'] = "CRUISE"
            df.loc[df['phase'].isna(), 'phase'] = 'DESCENT'
            df.drop(['squawk'], axis=1, inplace=True)
            df.dropna(inplace=True)
            if df.empty:
                continue

            df.to_pickle(f'{output_folder}{folder_name}/{inc}_{df.icao24.iloc[0]}.pkl')
            inc += 1

#%%

def get_dist_index(df_input, ref, ref_dist, side='left'):
    
    df = df_input.copy()
    
    if side == 'right':
        df = df.iloc[::-1]
    
    for i, row in df.iterrows():
        dist = vincenty((row.latitude, row.longitude), ref)
        if dist > ref_dist:
            return i-1

def distance_delta(input_df: pd.DataFrame) -> pd.Series:
    
    df = input_df.copy()
    df['lat_shifted'] = df['latitude'].shift(-1)
    df['lon_shifted'] = df['longitude'].shift(-1)

    delta = df.apply(lambda x: vincenty(
        (x['latitude'], x['longitude']), (x['lat_shifted'], x['lon_shifted'])), axis = 1)
    
    return delta

def get_distance(df, dep_airp, arr_airp):
    df.reset_index(drop=True, inplace=True)
    point_b = df[df['phase'] == 'CLIMB'].last_valid_index()
    point_c = df[df['phase'] == 'DESCENT'].first_valid_index()
    if point_b is None or point_c is None:
        return None, None
    dist_b = vincenty((df['latitude'].iloc[point_b], df['longitude'].iloc[point_b]), dep_airp)
    dist_c = vincenty((df['latitude'].iloc[point_c], df['longitude'].iloc[point_c]), arr_airp)
    return dist_b, dist_c

def phase_dist(filename, dep_airp, arr_airp):
    
    """
    Function to get the average phase point b and point c from past flights
    """
    
    df = pd.read_pickle(filename)
    d = df.groupby(('flight_id')).apply(get_distance, dep_airp, arr_airp)
    d = pd.DataFrame([[a,b] for a,b in d.values])
    d.dropna(inplace=True)
    return d
    
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
                df = feature_extraction(cleaned_traffic, eval(airport_refs[basename]))
                df.to_pickle(f'{output_folder}{basename}.pkl')
            elif output_type in ('bst', 'fdit'):
                bst_export(cleaned_traffic, basename, output_folder)
            
        

if __name__ == "__main__":
   main()
