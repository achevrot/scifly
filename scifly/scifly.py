#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:03:35 2020

@author: antoine
"""

import pandas as pd
import numpy as np
from vincenty import vincenty
import pyproj
import collections
import tensorflow as tf
    
def data_cleaner(input_df:pd.DataFrame, threshold=3):
    
    df = input_df.copy()
    
    df['lat_shift_c'] = df['latitude'].shift(-1)
    df['lon_shift_c'] = df['longitude'].shift(-1)
    df['lat_shift_f'] = df['latitude'].shift(100)
    df['lon_shift_f'] = df['longitude'].shift(100)
    
    df["delta_close"] = df.apply(
        lambda x: vincenty((x['latitude'], x['longitude']),
                           (x['lat_shift_c'], x['lon_shift_c'])), axis = 1)
    df["delta_far"] = df.apply(
        lambda x: vincenty((x['latitude'], x['longitude']),
                           (x['lat_shift_f'], x['lon_shift_f'])), axis = 1) 
    
    result = df[(df["delta_close"] < 30) & (df["delta_far"] < 100)]

    return result

def angle_distance(alpha, beta):
    phi = abs(beta - alpha) % 360
    distance = np.where(phi > 180, 360 - phi, phi)
    return distance

def track_delta(input_df: pd.DataFrame, endpoint_coordinate=None) -> pd.Series:
    
    geodesic = pyproj.Geod(ellps='WGS84')
    def calculate_ref(x_lat, x_lon, y_lat, y_lon):
        ref_s, back_azimuth, distance = geodesic.inv(x_lon, x_lat,
                                                     y_lon, y_lat)
        if ref_s < 0:
            ref_s += 360
        ref_e = (back_azimuth + 180) % 360
        return ref_s, ref_e
    
    if isinstance(endpoint_coordinate, tuple):
        track_ref = input_df.apply(lambda row:
                                   calculate_ref(row.latitude,
                                                 row.longitude,
                                                 endpoint_coordinate[0],
                                                 endpoint_coordinate[1])[0],
                                   axis=1)
        return angle_distance(input_df.track, track_ref)
    
    elif endpoint_coordinate is None:
        ref_s, ref_e = calculate_ref(input_df.latitude.iloc[0],
                                    input_df.longitude.iloc[0],
                                    input_df.latitude.iloc[-1],
                                    input_df.longitude.iloc[-1])
                                    

        track_ref = np.linspace(start=ref_s, stop=ref_e, num=len(input_df))
    
        return angle_distance(input_df.track, track_ref)

    else :
        raise RuntimeError(
                "Please enter a tuple (Latitude, Longitude) as endpoint_coordinate or leave empty"
            )
        
def flight_points(phases, phase_min = 10):
    counter = collections.Counter(phases)
    if (counter['DESCENT'] < phase_min or
            counter['CLIMB'] < phase_min or
            counter['CRUISE'] < phase_min):
        return None, None, None, None
    b = counter['CLIMB']
    c = len(phases) - counter['DESCENT']

    return 0,b,c,-1

def distance_delta(input_df: pd.DataFrame) -> pd.Series:
    
    df = input_df.copy()
    df['lat_shifted'] = df['latitude'].shift(-1)
    df['lon_shifted'] = df['longitude'].shift(-1)

    delta = df.apply(lambda x: vincenty(
        (x['latitude'], x['longitude']), (x['lat_shifted'], x['lon_shifted'])), axis = 1)
    
    return delta



