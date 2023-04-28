# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:55:59 2023

@author: Ivan
""" 

import pandas as pd
import os
import librosa as lr
import numpy as np
import librosa as lr


def load_data(path):
    
    train_df=pd.read_csv(os.path.join(path,'train.csv'))
    val_df=pd.read_csv(os.path.join(path,'val.csv'))
    train_df['file']=list(map(lambda i: os.path.join('data','train',str(i)+ '.wav') , train_df.index))
    val_df['file']=list(map(lambda i: os.path.join('data','val',str(i)+ '.wav') , val_df.index))
    
    return train_df, val_df
    
def signal_length(data):
    signal, sr = lr.load(data)
    signal = signal[:48000]
    zero_padding = np.zeros(48000 - signal.shape[0], dtype=np.float32)
    signal = np.concatenate((signal, zero_padding),0)
    return signal

def create_spectogram(df):
    df['spectogram']= list(map(lambda i: lr.amplitude_to_db(abs(lr.feature.melspectrogram(y=i, sr=22050, win_length=256, hop_length=512))) , df['wav']))
    return df



