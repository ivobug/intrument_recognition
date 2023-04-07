# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2023

@author: Ivan
"""

import pandas as pd
import os
from itertools import combinations
import random
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import glob



#Define path to train dataset
train_path='Dataset/IRMAS_Training_Data'

# here specify how many copies we want and specify the percentage for each of the n-combinations
TOTAL_SAMPLES=6000
ONE_COMB_PERCENTAGE=43
TWO_COMB_PERCENTAGE=44
THREE_COMB_PERCENTAGE=13

columns=['file','cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
df = pd.DataFrame( columns=columns)


for root, dirs, files in os.walk(train_path, topdown=True):
    for i in files:
        category=os.train_path.split(root)
        new_row = pd.Series({'file': os.train_path.join(root,i), category[1]:1})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

df=df.fillna(0)
#df.to_csv('out.csv')

files_list =df.index.values.tolist()

sep_files_list=[df[df[column]==1].index.values.tolist() for column in columns[1:]]


def createCombinations(listofLists, numOfElems, samples):
    listOfCombinations=[]
    for sampleOne in range(samples):
        oneComb=[]
        gen=random.sample(list(combinations(listofLists,numOfElems)), 1)[0]
        for i in gen:
            oneComb.append(random.sample(i,1)[0])
        listOfCombinations.append(oneComb)
    return listOfCombinations



#Sporo
#random.sample(list(product(*random.sample(list(combinations(sep_files_list,3)), 1)[0])),1)


one_elem_samples=random.choices(files_list, k=int(TOTAL_SAMPLES*(ONE_COMB_PERCENTAGE/100)))
one_elem_samples=list(map(lambda x: [x], one_elem_samples))

two_elem_samples=createCombinations(sep_files_list, 2,int(TOTAL_SAMPLES*TWO_COMB_PERCENTAGE/100))  

three_elem_samples=createCombinations(sep_files_list, 3,int(TOTAL_SAMPLES*THREE_COMB_PERCENTAGE/100))  

[one_elem_samples.extend(l) for l in (two_elem_samples,three_elem_samples)]


df_spectogram= pd.DataFrame( columns=columns)
#samples=[ two_elem_samples[:10]]

for combination in one_elem_samples:
    classes=[]
    audios=[]
    for file in combination:
        sample_org, sr1 = lr.load(df.loc[file]['file'],offset=False)
        audios.append(sample_org)
        classes.append(df.loc[file].eq(1).drop_duplicates().keys()[1])
    new_audio=sum(audios)
    listOfOnes = [1] * len(classes)
    ####################################################
    #add noise and convert to spectogram
    
    
    S = lr.feature.melspectrogram(y=new_audio, sr=22050, win_length=256, hop_length=512)
    Xdb_mel = [lr.amplitude_to_db(abs(S))]  
    
    
    ###################################################
    Xdb_mel.extend(listOfOnes)
    columns=['file']
    columns.extend(classes)
    new_row = pd.Series(Xdb_mel, index=columns)
    df_spectogram = pd.concat([df_spectogram, new_row.to_frame().T], ignore_index=True)
     
    
df_spectogram=df_spectogram.fillna(0)



def plot_images(rows,columns, inst=False, n_inst=False ):
    
    fig = plt.figure(figsize=(10, 7))
    ite=[x for x in range(rows*columns)] 
    temporary_df=df_spectogram
    if(n_inst!=False):
        temporary_df=df_spectogram[df_spectogram.drop(['file'], axis=1).sum(axis = 1)==n_inst]
    for i in ite:
        fig.add_subplot(rows, columns, i+1)
        if(inst!=False):
            listOfImages=temporary_df[temporary_df[inst]==1]['file'].tolist()
            plt.imshow(listOfImages[i])
            plt.title(f"{list(temporary_df[temporary_df[inst]==1].drop(['file'], axis=1).iloc[i][temporary_df[temporary_df[inst]==1].drop(['file'], axis=1).iloc[i].eq(1)].keys())}")
        else: 
            plt.imshow(df_spectogram['file'][i])
            plt.title(f"{list(temporary_df.drop(['file'], axis=1).loc[i][temporary_df.drop(['file'], axis=1).loc[i].eq(1)].keys())}")
        plt.axis('off')
        
    


def addNoise(noise_val):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    
    random_sample=df.sample()
    signal, sr = lr.load(random_sample['file'].iloc[0])
    soundfile.write('original.wav',signal,16000)
    
    S = lr.feature.melspectrogram(y=signal, sr=22050, win_length=256, hop_length=512)
    Xdb_mel = [lr.amplitude_to_db(abs(S))]  
    plt.imshow(Xdb_mel[0])
    plt.title(f'Original {list(random_sample.T[random_sample.eq(1).T].dropna().index)}')
    plt.axis('off')
    
    #RMS=math.sqrt(np.mean(signal**2))
    noise=np.random.normal(0, noise_val, signal.shape[0])
    signal_noise = signal+noise
    S = lr.feature.melspectrogram(y=signal_noise, sr=22050, win_length=256, hop_length=512)
    Xdb_mel = [lr.amplitude_to_db(abs(S))]  
    fig.add_subplot(1, 2, 2)
    plt.imshow(Xdb_mel[0])
    plt.title('Noise')
    plt.axis('off')
    
    soundfile.write('noise.wav',signal_noise,16000)


normal_noise_value=0.005


###################################################################
#---------------------------PLAY HERE------------------------------
###################################################################

#To run this function you have to input num of rows and cols,
# and if you want you can specify instrument and num of combinations

plot_images(2,2)
plot_images(2,2,'voi',1)
plot_images(2,2,'voi',2)

#This function will plot spectogram of random audio and same audio + noise
#Also it will save both audios in current folder
#You have to provide noise value
addNoise(0.010)


##################################################################
#Create Validation dataframe

val_path='Dataset/IRMAS_Validation_Data'
val_df = pd.DataFrame( columns=columns)

text_files = [f.split('\\')[-1] for f in glob.glob('Dataset/IRMAS_Validation_Data/*.txt')]
wav_suffix='wav'
txt_suffix='txt'


for file in text_files:
    fileName = os.path.splitext(file)[0]
    
    txt_path=os.path.join(val_path,fileName+ '.' + txt_suffix)
    lines = open(txt_path, 'r').readlines()
    classes=[x.strip() for x in lines]
    columns=['file']
    columns.extend(classes)
    
    signal, sr = lr.load(os.path.join(val_path,fileName+ '.' + wav_suffix))
    S = lr.feature.melspectrogram(y=new_audio, sr=22050, win_length=256, hop_length=512)
    Xdb_mel = [lr.amplitude_to_db(abs(S))]
    listOfOnes = [1] * len(classes)
    Xdb_mel.extend(listOfOnes)
    
    new_row = pd.Series(Xdb_mel, index=columns)
    val_df = pd.concat([val_df, new_row.to_frame().T], ignore_index=True)
    
val_df=val_df.fillna(0)

#Add extra column to train and validation data
val_df['hot_enc']=(val_df.drop(['file'],1).values).tolist()
df_spectogram['hot_enc']=(df_spectogram.drop(['file'],1).values).tolist()

#save Dataframes to csv
os.makedirs('data', exist_ok=True)  
val_df.to_csv('data/train.csv')  
df_spectogram.to_csv('data/val.csv') 
