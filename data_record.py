#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:03:33 2020

@author: antoine Chevrot
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import tensorflow as tf

class DataRecord():
    """
    
    DataRecord is an helper class to parse flight tfrecords into datasets
    usable for training. 
    
    """
    def __init__(self, filenames, sequence_features):
        self.sequence_features = sequence_features
        self.dataset = self.read_record(filenames)

    def _parse_function(self, example_proto):

        # Define features
        context_features = {
            'icao': tf.io.FixedLenFeature([], dtype=tf.string),
            'cal': tf.io.FixedLenFeature([], dtype=tf.string)
            }
        
        # Parsing the records
        context, sequence = tf.io.parse_single_sequence_example(
            example_proto, context_features, self.sequence_features)
        
        # context
        icao = tf.cast(context['icao'], dtype = tf.string)
        callsign = tf.cast(context['cal'], dtype = tf.string)
    
        # features
        data = tf.transpose(tf.cast([v for k,v in sequence.items() 
                                     if k not in ('label')], dtype=tf.float32))
        label = tf.cast(sequence['label'], dtype=tf.int32)
        
        return icao, callsign, data, label
    
    def windowed_dataset(self, window_size, batch_size, shuffle_buffer):
        dataset = self.dataset.map(lambda icao, callsign, data, label : data)
        dataset = dataset.unbatch()
        dataset = dataset.window(window_size, 1, drop_remainder=True)
    
        def create_sequence_ds(chunk):
            return chunk.batch(window_size, drop_remainder=True)
        
        dataset = dataset.flat_map(create_sequence_ds)
        dataset = (dataset.shuffle(shuffle_buffer)
                    .batch(batch_size, drop_remainder=True).prefetch(1))
        return dataset
    
    @tf.autograph.experimental.do_not_convert
    def get_data(self, windowed=True, window_size=None, batch_size=None,
                 shuffle_buffer=None):
        if windowed:
            if window_size is None:
                window_size = 30
            if batch_size is None:
                batch_size = 128
            if shuffle_buffer is None:
                shuffle_buffer = 1024
            dataset = self.windowed_dataset(
                window_size, batch_size, shuffle_buffer)
        else:
            dataset = self.dataset.map(
                lambda icao, callsign, data, label : data) 
        return dataset
    
    @tf.autograph.experimental.do_not_convert
    def get_callsigns(self):        
        cal_dataset = self.dataset.map(
            lambda icao, callsign, data, label: callsign)
        callsigns = [tensor.numpy().decode() for tensor in cal_dataset]
        return callsigns
    
    @tf.autograph.experimental.do_not_convert
    def get_icaos(self):
        icao_dataset = self.dataset.map(
            lambda icao, callsign, data, label: icao)
        icaos = [tensor.numpy().decode() for tensor in icao_dataset]
        return icaos
    
    @tf.autograph.experimental.do_not_convert
    def get_labels(self):
        lab_dataset = self.dataset.map(
            lambda icao, callsign, data, label: label)
        labels = [tensor.numpy() for tensor in lab_dataset]
        return labels

   
    def read_record(self, filenames):
        files = tf.data.Dataset.list_files([filenames])
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=3,
            num_parallel_calls=tf.data.experimental.AUTOTUNE) 
        dataset = dataset.map(self._parse_function, num_parallel_calls=2)
        return dataset