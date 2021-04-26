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

def svdd_prediction(model1_svdd,model2_svdd,test_data_path,test_folders,Rvar1,cvar1,Rvar2,cvar2):
    print("==============PREDICTING THE LABELS ==============================")
    X_validate =  load_training_images(test_data_path, test_folders)
    X_validate = np.array(X_validate)
    X_validate = np.reshape(X_validate, [-1, X_validate.shape[1],X_validate.shape[2],X_validate.shape[3]])
    anomaly=0
    anomaly1=0
    for i in range(0,len(X_validate)):
        img = np.array(X_validate[i])[np.newaxis]
        predicted_reps = model1_svdd.predict(img)
        dist1 = np.sum(((predicted_reps - cvar1) ** 2), axis=1)
        if(dist1>Rvar1): #where 10.0 is the threshold.
            anomaly+=1
        predicted_reps = model2_svdd.predict(img)
        dist2 = np.sum(((predicted_reps - cvar2) ** 2), axis=1)
        if(dist2>Rvar2): #where 10.0 is the threshold.
            anomaly1+=1
    print("SVDD1 identified:%d anomalies"%anomaly)
    print("SVDD2 identified:%d anomalies"%anomaly1)

if __name__ == '__main__':
    test_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    test_folders = ["clear-day-70"]
    model1_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/model-weights-rain/" #path to save the svdd weights
    model2_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/model-weights-light/" #path to save the svdd weights
    cvar1 = np.load(model1_path+"svdd_center.npy")
    Rvar1 = np.load(model1_path+"svdd_radius.npy")
    cvar2 = np.load(model2_path+"svdd_center.npy")
    Rvar2 = np.load(model2_path+"svdd_radius.npy")
    model1_svdd=load_model(model1_path)
    model2_svdd=load_model(model2_path)
    svdd_prediction(model1_svdd,model2_svdd,test_data_path,test_folders,Rvar1,cvar1,Rvar2,cvar2)
