# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:39:10 2020

@author: Nitika


"""

#importing required libraries
import numpy as np
import pandas as pd

import os
import librosa

import scipy
from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas

tqdm.pandas()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#Loading the labels dataset and audio files

audio_train_files = os.listdir('C:/Users/Nitika/Audio Recognition - Cough Analysis/input/audio_train')
audio_test_files = os.listdir('C:/Users/Nitika/Audio Recognition - Cough Analysis/input/audio_test')

train = pd.read_csv('C:/Users/Nitika/Audio Recognition - Cough Analysis/input/train.csv')
submission = pd.read_csv('C:/Users/Nitika/Audio Recognition - Cough Analysis/input/sample_submission.csv')

#cleaning the dataset and extracting features
SAMPLE_RATE = 44100

def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

# Generate mfcc features with mean and standard deviation
def get_mfcc(name, path):
    data, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    except:
        print('bad file')
        return pd.Series([0]*210)
	
def convert_to_labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

#Data Preparation
train_data = pd.DataFrame()
train_data['fname'] = train['fname']
test_data = pd.DataFrame()
test_data['fname'] = audio_test_files

train_data = train_data['fname'].progress_apply(get_mfcc, path='C:/Users/Nitika/Audio Recognition - Cough Analysis/input/audio_train/')
print('done loading train mfcc')
test_data = test_data['fname'].progress_apply(get_mfcc, path='C:/Users/Nitika/Audio Recognition - Cough Analysis/input/audio_test/')
print('done loading test mfcc')

train_data['fname'] = train['fname']
test_data['fname'] = audio_test_files

train_data['label'] = train['label']
test_data['label'] = np.zeros((len(audio_test_files)))

train_data.head()

#Combining the features together in a dataset

X = train_data.drop(['label', 'fname'], axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train_data.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])

X_test = test_data.drop(['label', 'fname'], axis=1)
X_test = X_test.values

print(X.shape)
print(X_test.shape)

#Using Pca for dimension reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=65).fit(X_scaled)
X_pca = pca.transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(sum(pca.explained_variance_ratio_)) 

#########################################################################
##################################SVM Model##############################
#########################################################################

#Fitting SVM Model
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size = 0.2, random_state = 42, shuffle = True)

clf = SVC(kernel = 'rbf', C = 4, gamma = 0.01, probability=True)

clf.fit(X_train, y_train)

print(accuracy_score(clf.predict(X_val), y_val))

#Fitting the entire training set and predicting output

clf.fit(X_pca, y)
str_preds = clf.predict(X_test_pca)
subm = pd.DataFrame()
subm['fname'] = audio_test_files
subm['label'] = str_preds
subm.to_csv('SVMpredictions.csv', index=False)

#############################################################################
##################################Neural Network#############################
#############################################################################

# defining the keras model
model = Sequential()
model.add(Dense(12, input_dim=65, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compiling the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=10)

# evaluating the model
accuracy = model.evaluate(X_val, y_val)
print('Accuracy: %.2f' % (accuracy[1]*100))


predictions = model.predict_classes(X_test_pca)
subm = pd.DataFrame()
subm['fname'] = audio_test_files
subm['label'] = predictions
subm.to_csv('NNpredictions.csv', index=False)