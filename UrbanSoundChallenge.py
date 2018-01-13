#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:36:49 2018

@author: jarvis
"""
DATA_ROOT = '/home/jarvis/Downloads/UrbanSoundChallenge'
audio_path = DATA_ROOT + '/train/Train'
pict_Path = DATA_ROOT + '/train/Train-pict'

import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib
import matplotlib as plt

def wav2img (wav_path, targetdir, figsize=(4,4)):
    wavefile, samplerate = sf.read(wav_path)
    wavefile = np.mean(wavefile, axis=1)
    plt.pyplot.specgram(wavefile, NFFT=1024, Fs=samplerate, noverlap=512)
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    matplotlib.pyplot.savefig('%s.png' % output_file)
  

for i in range (0,8730):
    try:
        wav2img(audio_path + '/' + str(i) + '.wav', pict_Path)
    except Exception:
        pass

test_audio_path = DATA_ROOT + '/test/Test'
test_pict_Path = DATA_ROOT + '/test/Test-pict'
    
for i in range (0,8730):
    try:
        wav2img(test_audio_path + '/' + str(i) + '.wav', test_pict_Path)
    except Exception:
        pass
    
train = pd.read_csv('train_fuSp8nd.csv')
test = pd.read_csv('test_B0QdNpj.csv')

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 10, activation='softmax'))

classifier.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=["accuracy"])

import os
lst = os.listdir(pict_Path)
for element in lst:
    element = element.split('.')[0]
    temp = train[train['ID'] == int(element)].index.tolist()
    train.loc[temp,2] = 'Y'
    
train = train[train[2] == 'Y']

#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#encoder.fit(train['Class'])
#encoded_Y = encoder.transform(train['Class'])
#from keras.utils import np_utils
#dummy_y = np_utils.to_categorical(encoded_Y)
      
## AFTER PUTING THE IMAGE IN THERE RESPECTIVE SUB FOLDERS ## 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2, width_shift_range=0.4)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(pict_Path, target_size=(150, 150), batch_size=32, class_mode='binary')

classifier.fit_generator(train_generator, steps_per_epoch=200, epochs=50)
