#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:03:35 2020

@author: antoine
"""

import pandas as pd
from vincenty import vincenty
from datetime import timedelta

def data_cleaner(input_df:pd.DataFrame):
    
    df = input_df.copy()
    
    df['lat_shift_1'] = df['latitude'].shift(-1)
    df['lon_shift_1'] = df['longitude'].shift(-1)
    df['lat_shift_30'] = df['latitude'].shift(300)
    df['lon_shift_30'] = df['longitude'].shift(300)
    
    df["delta_close"] = df.apply(
        lambda x: vincenty((x['latitude'], x['longitude']),
                           (x['lat_shift_1'], x['lon_shift_1'])), axis = 1)
    df["delta_far"] = df.apply(
        lambda x: vincenty((x['latitude'], x['longitude']),
                           (x['lat_shift_30'], x['lon_shift_30'])), axis = 1) 
    
    r_c = df['delta_close'].rolling('300s')
    r_f = df['delta_far'].rolling('300s')
    
    result = df[((df.delta_close - r_c.median()).abs() < 3 * r_c.std())
                & ((df.delta_far - r_f.median()).abs() < 3 * r_f.std())]

    return result