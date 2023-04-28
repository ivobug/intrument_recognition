# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:15:47 2023

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
import soundfile as sf




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
        category=os.path.split(root)
        new_row = pd.Series({'file': os.path.join(root,i), category[1]:1})
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


df_train= pd.DataFrame( columns=columns)
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
    new_audio=[sum(audios)]
    new_audio.extend(listOfOnes)
    columns=['file']
    columns.extend(classes)
    new_row = pd.Series(new_audio, index=columns)
    df_train = pd.concat([df_train, new_row.to_frame().T], ignore_index=True)
     
    
df_train=df_train.fillna(0)

        
    


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
    Signal=[signal]
    listOfOnes = [1] * len(classes)
    Signal.extend(listOfOnes)
    
    new_row = pd.Series(Signal, index=columns)
    val_df = pd.concat([val_df, new_row.to_frame().T], ignore_index=True)
    
val_df=val_df.fillna(0)

#Add extra column to train and validation data
val_df['hot_enc']=(val_df.drop(['file'],1).values).tolist()
df_train['hot_enc']=(df_train.drop(['file'],1).values).tolist()


#save Dataframes to csv
def save_data(df, name):
    os.makedirs(f'data/{name}', exist_ok=True)
    
    for i in range(len(df['file'])):
        sf.write(f'data/{name}/{i}.wav', df['file'].iloc[i], 22050, 'PCM_24')

    df=df.drop(['file'],axis=1)
    df.to_csv(f'data/{name}.csv') 
    
save_data(df_train, 'train')
save_data(val_df, 'val')
