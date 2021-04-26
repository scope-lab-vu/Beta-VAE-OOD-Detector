#!/usr/bin/env python
# coding: utf-8
import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
import psutil
from sklearn.utils import shuffle
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.activations import linear
from keras.models import Model, model_from_json
import numpy as np
from keras.callbacks import Callback, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

def load_model(model_path):
    with open(model_path + 'svdd_architecture.json', 'r') as jfile:
            model_svdd = model_from_json(jfile.read())
    model_svdd.load_weights(model_path + 'svdd_weights.h5')
    return model_svdd

#Load complete input images without shuffling
def load_images(paths):
    numImages = 0
    inputs = []
    for path in paths:
        numFiles = len(glob.glob1(path,'*.png'))
        numImages += numFiles
        for img in glob.glob(path+'*.png'):
            img = cv2.imread(img)
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            inputs.append(img)
    #inpu = shuffle(inputs)
    print("Total number of images:%d" %(numImages))
    return inputs

def createFolderPaths(train_data_path, train_folders):
    paths = []
    for folder in train_folders:
        path = train_data_path + folder + '/'
        paths.append(path)
    return paths

def load_training_images(train_data_path, train_folders):
    paths = createFolderPaths(train_data_path, train_folders)
    return load_images(paths)

#Load complete input images without shuffling
def load_training_images1(train_data):
    inputs = []
    comp_inp = []
    with open(train_data + 'calibration.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img = cv2.imread(train_data + row[0])
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                inputs.append(img)
            return inputs


def svdd_prediction(model_svdd,test_data_path,test_folders,Rvar,cvar):
    print("==============PREDICTING THE LABELS ==============================")
    X_validate =  load_training_images(test_data_path,test_folders)
    X_validate = np.array(X_validate)
    X_validate = np.reshape(X_validate, [-1, X_validate.shape[1],X_validate.shape[2],X_validate.shape[3]])
    anomaly=0
    score_val=[]
    for i in range(0,len(X_validate)):
        val=[]
        anomaly_val=0
        img = np.array(X_validate[i])[np.newaxis]
        predicted_reps = model_svdd.predict(img)
        dist = np.sum(((predicted_reps - cvar) ** 2), axis=1)
        scores = dist - Rvar
        score_val.append(scores)
        print(scores)
        cpu = psutil.cpu_percent()
        #print(scores)
        if(scores> 2.8): #where 10.0 is the threshold.
            anomaly_val=1
            anomaly+=1
        # gives an object with many fields
        #mem = psutil.virtual_memory().total / (1024.0 ** 3)
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss/ (1000.0 ** 3)
        val.append(anomaly_val)

        with open('/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/SVDD/svdd-illu-rain.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(val)

    print(anomaly)
    print(max(score_val))

if __name__ == '__main__':
    test_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/new/dataset/"   #"/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"#"/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/data-generator/"       #"/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    test_folders = ["heavy-rain-70"]
    model_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/SVDD/train-illumination/" #path to save the svdd weights
    cvar = np.load(model_path+"svdd_center.npy")
    Rvar = np.load(model_path+"svdd_radius.npy")
    model_svdd=load_model(model_path)
    svdd_prediction(model_svdd,test_data_path,test_folders,Rvar,cvar)
