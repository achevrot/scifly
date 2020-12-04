#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:51:07 2020

@author: antoine
"""
import tensorflow as tf
from ../scifly import DataRecord


sequence_features = {
            'alt': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'd_a': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'd_b': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'd_c': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'd_d': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'hdg_delta': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'label': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'lat': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'lon': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
            'spd': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        }



def test_getters():
    record = DataRecord("tests/test_data/WZZ1A_test.tfrecord", sequence_features)
    assert isinstance(record.get_icaos(), list)
