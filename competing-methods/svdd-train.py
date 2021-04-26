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
seed_value = 56
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


#GPU access
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

nu = 0.1

#Load complete input images without shuffling
def load_images(path):
        numImages = 0
        inputs = []
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


class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.radius = radius
        self.model = model
        self.inputs = X_train
        self.cvar = cvar
        self.y_reps = np.zeros((len(X_train), 4))

    #At the end of each epoch, compute a new radius.
    #Use the distance of the predictions from the center
    #Compute the new radius as the qth quantile

    def on_epoch_end(self, batch, logs={}):

        reps = self.model.predict(self.inputs)#predictions
        self.y_reps = reps
        center = self.cvar
        dist = np.sum((reps - self.cvar) ** 2, axis=1)#compute distance of predictions from center
        scores = dist
        val = np.sort(scores)
        R_new = np.percentile(val, nu * 100)  # qth quantile of the radius.
        self.radius = R_new
        print("[INFO:] \n Updated Radius Value...", R_new)
        return self.radius

def create_model(inputs):
        input_img = Input(shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        x = Conv2D(32, (5, 5),  use_bias=False, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(128, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(256, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(1568)(x)
        x = linear(x)

        model = Model(input_img, x)
        #model.summary()
        return model

def initialize_c_with_mean(inputs, model,cvar,Rvar):
        reps = model.predict(inputs)
        print(len(reps[0]))

        eps = 0.1
        c = np.mean(reps, axis=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        cvar = c

        dist = np.sum((reps - c) ** 2, axis=1)
        val = np.sort(dist)
        Rvar = np.percentile(val, nu * 100)

        print("Radius initialized.", Rvar)

        return cvar, Rvar


# Custom loss SVDD_loss ball interpretation
def custom_ocnn_hypershere_loss(cvar):

    center = cvar

    # define custom_obj_ball
    def custom_obj_ball(y_true, y_pred):
        # compute the distance from center of the circle to the

        dist = (K.sum(K.square(y_pred - center), axis=1))
        avg_distance_to_c = K.mean(dist)

        return (avg_distance_to_c)

    return custom_obj_ball

def custom_ocnn_hyperplane_loss(center, r):

    def custom_hinge(y_true, y_pred):

        term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
        term3 = 1 / nu * K.mean(term3)
        loss = term3
        return (loss)

    return custom_hinge


def fit(bound,X_train,save_path):
    inputs = X_train #input data
    Rvar = 1.0 #radius of the circle
    cvar = 0.0 #center of the circle
    model_svdd = create_model(inputs)  #Creating the model
    cvar, Rvar = initialize_c_with_mean(inputs, model_svdd,cvar,Rvar) #initialize the radius of the circle

    out_batch = Adjust_svdd_Radius(model_svdd, cvar, Rvar, inputs) #Keep updating the radius of the circle

    def lr_scheduler(epoch): #learningrate scheduler to adjust learning rate.
        lr = 1e-5
        if epoch > 150:
            lr = 1e-6
            if(epoch == 151):
                print('lr: rate adjusted for fine tuning %f' % lr)
        return lr

    scheduler = LearningRateScheduler(lr_scheduler)
    opt = Adam(lr=1e-4) #choosing adam optimizer
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(save_path+'best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [out_batch, scheduler, es, mc]
    if bound == "soft_boundary":
        model_svdd.compile(loss=custom_ocnn_hyperplane_loss(cvar, out_batch.radius.astype(np.float32)), optimizer=opt)
    else:
        model_svdd.compile(loss=custom_ocnn_hypershere_loss(cvar), optimizer=opt)
    y_reps = out_batch.y_reps
    Rvar = out_batch.radius

    model_svdd.fit(inputs, y_reps, shuffle=True, batch_size=16, epochs=250, validation_split=0.1, verbose=1, callbacks=callbacks)
    Rvar = out_batch.radius
    cvar = out_batch.cvar
    print(cvar, Rvar)

    return model_svdd, cvar, Rvar

def save_model(path,model_svdd):
    model_svdd.save_weights(os.path.join(path, "svdd_weights.h5"))
    with open(os.path.join(path, 'svdd_architecture.json'), 'w') as f:
        f.write(model_svdd.to_json())
    np.save(os.path.join(path, 'svdd_center'), cvar)
    np.save(os.path.join(path, 'svdd_radius'), Rvar)



def svdd_prediction(test_data_path,test_folders,Rvar,cvar):
    print("==============PREDICTING THE LABELS ==============================")
    X_validate =  load_training_images(test_data_path, test_folders)
    X_validate = np.array(X_validate)
    X_validate = np.reshape(X_validate, [-1, X_validate.shape[1],X_validate.shape[2],X_validate.shape[3]])
    predicted_reps = model_svdd.predict(X_validate)
    dist = np.sum(((predicted_reps - cvar) ** 2), axis=1)
    scores = dist #- Rvar
    print(scores)
    anomaly=0
    for i in range(len(scores)):
        if(scores[i]>Rvar): #where 10.0 is the threshold.
            anomaly+=1
    print(anomaly)

#Load complete input images without shuffling
def load_training_images(train_data):
    inputs = []
    comp_inp = []
    with open(train_data + 'train.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img = cv2.imread(train_data + row[0])
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                inputs.append(img)
            for i in range(0,len(inputs),3):
                    comp_inp.append(inputs[i])
            print("Total number of images:%d" %len(comp_inp))
            return inputs, comp_inp


if __name__ == '__main__':
    train_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/data-generator/illumination/"
    #train_folders = ["clear-day","clear-day-50","evening","evening-50","mild-rain","mild-rain-50"]   #light - clear-day, clear-day-50      #rain - clear-day, mild-rain
    #test_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    #test_folders = ["heavy-rain"]
    #comp_inp,csv_input = load_training_images(train_data)
    save_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/SVDD/train-illumination/" #path to save the svdd weights
    bound = "soft_bound"
    #Loading images from the datasets
    csv_input = load_images(train_data_path)
    csv_input = shuffle(csv_input)

    #Split the data to train and test. Then shuffle it for training
    X_train, X_test = np.array(csv_input[0:len(csv_input)-0].copy()), np.array(csv_input[len(csv_input)-0:len(csv_input)].copy())
    X_train = np.reshape(X_train, [-1, X_train.shape[1],X_train.shape[2],X_train.shape[3]])
    #X_test = np.reshape(X_test, [-1, X_test.shape[1],X_test.shape[2],X_test.shape[3]])
    #X = (X_train, X_test)
    model_svdd, cvar, Rvar = fit(bound, X_train,save_path)
    save_model(save_path,model_svdd)
    svdd_prediction(test_data_path,test_folders,Rvar,cvar)
