#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:14:51 2020

@author: antoine chevrot
"""
from traffic.data import opensky
from datetime import timedelta
from traffic.core import loglevel
import pandas as pd
# from scifly.fdit import sbs_converter
# import scifly
from traffic.core.traffic import Traffic, Flight
import matplotlib.pyplot as plt
import geopandas
import tensorflow as tf

loglevel('INFO')

# %%

# Checking flight from London to warsovie 1 Sept to 15 Sept
flight_list = opensky.flightlist(start=1598961600,
                                 stop=1600171200,
                                 departure_airport='EGGW',
                                 arrival_airport='EPWA'
                                 )

print(flight_list)
# %%

# Checking flight from Madrid to Moscow from 1 Sept to 30 Sept
flight_list = opensky.flightlist(start=1598961600,
                                 stop=1600171200,
                                 departure_airport='LEMD',
                                 arrival_airport='UUEE'
                                 )



print(flight_list.head())
# %%

# Selecting only DF17 and DF18 database
my_raw_tables = [
    "identification_data4",
    "operational_status_data4",
    "position_data4",
    "velocity_data4",
    ]

# %%
flights = flight_list.callsign.unique()

# War to Lon flight in early September
df = opensky.rawdata(
    1598961600, # start_date
    1600171200, # end_date
    date_delta=timedelta(days=1), # change the number of requests
    table_name = my_raw_tables, # to filter to df17 and df18
    departure_airport='EGGW',
    arrival_airport='EPWA',
    cached=True
    )


df_decode = df.decode()
df_decode = df_decode.assign_id().eval()
df_decode = df_decode.drop_duplicates()
df_decode.data.onground.fillna(False, inplace=True)
df_decode = df_decode.__class__(df_decode.data.loc[~df_decode.data.onground])
df_decode.to_pickle('/home/antoine/Documents/Lon_War_1_15_Sept.pkl')

#%%

# df_decode = Traffic(pd.read_pickle('/home/antoine/Documents/Mad_Mos_1_15_Sept.pkl'))
df_brut = Traffic.from_file('/home/antoine/Documents/Lon_War_1_15_Sept.pkl')

# %%

# This is where you can downsample the data to have e.g. 1 msg every 2s :

df_decode = df_brut.resample('2s').eval()

# And then we clean the data after setting the ts index :
    
df = df_decode.data.copy()
df = df.set_index('timestamp')
df_cleaned = df.groupby(by='flight_id').apply(data_cleaner)
df_cleaned.drop(['lat_shift_c','lat_shift_f','lat_shift_c','onground', 'lon_shift_f', 'lon_shift_c', 'flight_id'], axis=1, inplace=True)
df_cleaned.reset_index(inplace=True)

# %%

# Plotting the cleaned traffic data :

clean_traffic = Traffic(df_cleaned)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
europe = world.query('continent == "Europe"')
i = clean_traffic.iterate()

#%%
flight = next(i) 
ax = europe.plot(color='white', edgecolor='black', figsize=(80,10))
ax.set_xlim(-10, 45)
ax.set_ylim(30, 80)
flight.plot(ax)
# df_decode.plot(ax)
plt.show()
data = flight.data

# %%
input_df = flight.data.iloc[:500]
ref, track = track_delta(flight.data)
df_cleaned
# %%

flight_phase = flight.phases()
a,b,c,d = flight_points(flight_phase.data.phdef phase(df_inp: pd.DataFrame):
    
    df = df_inp.copy()
    
    df['phase'] = fp.fuzzylabels(df['ts'], df['alt'],
                                 df['spd'], df['roc'], twindow=60)

    return dfase)

for i in range(4):df_cleaned
    index = (a, b, c, d)[i]
    pos = (df["lon"].iloc[index], df["lat"].iloc[index])
    dist = [vincenty(pos, (lon, lat))
            for (lon, lat) in zip(df["lon"], df["lat"])]
    df[f"d_{('a', 'b', 'c', 'd')[i]}"] = dist
    
    df.apply(
        lambda x: vincenty((x['latitude'], x['longitude']),
                           (x['lat_shift_f'], x['lon_shift_f'])), axis = 1) 

# %%

# For an export to FDI-T :
    
inc = 1 # For naming the files
for flight in clean_traffic:
    sbs_data = sbs_converter(flight.data)
    sbs_data.to_csv(f'/home/antoine/Documents/test/{inc}_{flight.data.icao24.iloc[0]}.bst', index=False, header=False)
    inc += 1


#%%

# Next step is to transform into a tfrecord for TF model training
# We first need to define and get all the features we want for our model :
    
# In this example we will use :
    
    # altitude : original data (OD)
    # speed : OD
    # longitude : OD
    # latitude : OD
    # track_delta : difference between real track and optimal track
    # delta : distance in km between 2 consecutive messages
    # phase : the flight phase the message is in
    # label : legit or modified message (0 or 1)
    
# %%

# Cleaning extra useless columns
df_cleaned.drop(["start","stop","track_unwrapped", "delta_close", "delta_far", "geoaltitude"], axis=1, inplace=True)

# Track delta

track_delta = df_cleaned.groupby('flight_id').apply(track_delta)
df_cleaned['track_delta'] = np.concatenate(list(track_delta.values))

# delta

df_cleaned['delta'] = df_cleaned.groupby('flight_id').apply(distance_delta).values

# phase

df_cleaned['phase'] = df_cleaned.groupby("flight_id").apply(lambda flight : Flight(flight).phases().data.phase).values
df_cleaned['phase'] = df_cleaned.phase.replace('NA', None) #replace NA and LEVEL by None and automatically pad the serie
df_cleaned['phase'] = df_cleaned.phase.replace('LEVEL', None)

# label

df_cleaned['label'] = 0

# Final check to get rid of nan values that will disturb training

df_cleaned.dropna(inplace=True)

# %%

# Serializing

# Helper functions for the serialisation :

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
      'icao': _bytes_feature(data.icao24.iloc[0].encode()),
      'cal': _bytes_feature(data.callsign.iloc[0].encode())
    })

    alt_features = data.altitude.apply(_float_feature).values
    spd_features = data.groundspeed.apply(_float_feature).values
    delta_features = data.delta.apply(_float_feature).values
    lat_features = data.latitude.apply(_float_feature).values
    lon_features = data.longitude.apply(_float_feature).values
    track_delta_features = data.track_delta.apply(_float_feature).values
    phase_features = data.phase.str.encode("utf-8").apply(_bytes_feature).values
    label_features = data.label.apply(_float_feature).values
    
    alt = tf.train.FeatureList(feature=alt_features)
    spd = tf.train.FeatureList(feature=spd_features)
    delta = tf.train.FeatureList(feature=delta_features)
    lat = tf.train.FeatureList(feature=lat_features)
    lon = tf.train.FeatureList(feature=lon_features)
    track_delta = tf.train.FeatureList(feature=track_delta_features)
    phase = tf.train.FeatureList(feature=phase_features)
    label = tf.train.FeatureList(feature=label_features)

    messages = tf.train.FeatureLists(feature_list={
      'alt': alt,
      'delta': delta,
      'label': label,
      'lat': lat,
      'lon': lon,
      'phase': phase,
      'track_delta': track_delta,
      'spd': spd,
      })

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.SequenceExample(context=flight,
                                             feature_lists=messages)

    return example_proto.SerializeToString()


# ----------------------------------------

serialized_data = df_cleaned.groupby('flight_id').apply(serialize_example)


# Writing into tfrecord files

for index, s_data in serialized_data.items():
    with tf.io.TFRecordWriter(f'/home/antoine/Documents/Test_data/training_test/{index}_test.tfrecord') as writer:
        writer.write(s_data)


# %%

# Checking the readability of the file :
    
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
    
record = DataRecord("/home/antoine/Documents/Test_data/training_test/*tfrecord", sequence_features)
data = record.get_data()
t = record.get_phases()
