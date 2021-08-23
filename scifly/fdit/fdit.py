#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:07:59 2020

@author: antoine
"""

import pandas as pd
import numpy as np


def sbs_converter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a pandas dataframe with flight information and convert it into a
    pandas dataframe in SBS-like format readable by FDI-T platform.

    Parameters
    ----------
    data : pd.DataFrame
        Must include following data :
            mintime : time data for each message ;
            icao24 : aircraft ID ;
            callsign ;
            speed ;
            altitude ;
            vertical rate ;
            longitude ;
            latitude ;
            heading ;

    Returns
    -------
    sbs_data : pd.DataFrame
        Return a SBS-like pandas dataframe. Call the pd.DataFrame.to_csv()
        function on it to get your .BST file. Don't forget to set index and
        header parameter to False.

    """

    sbs_data = df.copy()
    sbs_data['msg_type'] = 3
    sbs_data["msg"] = "MSG"
    sbs_data["msg_type2"] = "3"
    
    icao = [
        value
        for value in ["icao", "icao24"]
        if value in list(sbs_data.columns)
    ]
    sbs_data["icao24"] = sbs_data[icao[0]].str.upper()
    sbs_data["icao24_dec"] = sbs_data.icao24.apply(int, base=16)

    sbs_data["icao24_2"] = sbs_data.icao24_dec

    mintime = [
        value
        for value in ["mintime", "ts", 'timestamp']
        if value in list(sbs_data.columns)
    ]
    
    sbs_data["date"] = pd.to_datetime(sbs_data[mintime[0]],
                                      unit="s").dt.strftime("%Y/%m/%d")
    sbs_data["time"] = (
        pd.to_datetime(sbs_data[mintime[0]], unit="s")
        .dt.strftime("%H:%M:%S.%f")
        .str.slice(0, -3, 1)
    )

    sbs_data["date_2"] = sbs_data["date"]
    sbs_data["time_2"] = sbs_data["time"]

    alt = [
        value
        for value in ["altitude", "alt"]
        if value in list(sbs_data.columns)
    ]
    sbs_data["alt"] = sbs_data[alt]  # *3.28084

    sbs_data["alt"] = sbs_data.alt.round().astype("Int64")

    vel = [
        value
        for value in ["vel", "velocity", "speed", "spd", "groundspeed"]
        if value in list(sbs_data.columns)
    ]
    sbs_data["velocity"] = sbs_data[vel]  # *1.94384

    lat = [
        value
        for value in ["latitude", "lat"]
        if value in list(sbs_data.columns)
    ]

    lon = [
        value
        for value in ["longitude", "lon"]
        if value in list(sbs_data.columns)
    ]

    vert = [
        value
        for value in ["vertrate", "vertical_rate", "roc"]
        if value in list(sbs_data.columns)
    ]
    
    heading = [
        value
        for value in ["hdg", "heading", "track"]
        if value in list(sbs_data.columns)
    ]
    
    callsign = [
        value
        for value in ["callsign", "cal", 'cals']
        if value in list(sbs_data.columns)
    ]
    

    sbs_data["squawk"] = np.nan
    sbs_data.loc[sbs_data["msg_type"] == 3, "alert"] = "0"
    sbs_data.loc[sbs_data["msg_type"] == 3, "emergency"] = "0"
    sbs_data.loc[sbs_data["msg_type"] == 3, "spi"] = "0"
    sbs_data.loc[sbs_data["msg_type"] == 3, "surface"] = "0"

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
        callsign[0],
        alt[0],
        vel[0],
        heading[0],
        lat[0],
        lon[0],
        vert[0],
        "squawk",
        "alert",
        "emergency",
        "spi",
        "surface",
    ]

    sbs_data = sbs_data[cols]
    sbs_data.sort_values(by=["date", "time", "icao24"], inplace=True)

    return sbs_data


def from_fdit(filename) -> pd.DataFrame:
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

    sbs_data_ano = pd.read_csv(filename, names=cols)
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
