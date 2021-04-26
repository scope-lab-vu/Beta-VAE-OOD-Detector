#!/usr/bin/env python3
#libraries
import time
import random
import csv
import cv2
import os
import glob
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam,SGD
from keras.models import model_from_json, load_model
from keras.layers import Input, Dense
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Lambda, Input, Dense, MaxPooling2D, BatchNormalization,Input
from keras.layers import UpSampling2D, Dropout, Flatten, Reshape, RepeatVector, LeakyReLU,Activation
from keras.callbacks import ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import History
from itertools import product
import matplotlib.pyplot as plt
import prettytable
from prettytable import PrettyTable
from sklearn.utils import shuffle
# seed_value = 56
# os.environ['PYTHONHASHSEED']=str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"#Setting the script to run on GPU:1,2
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#GPU access
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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


#Create the Beta-VAE model
def CreateModels(nl, b):
    #sampling function of the Beta-VAE
    def sample_func(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    model = Sequential()
    input_img = Input(shape=(224,224,3), name='image')
    x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.1)(x)
    #x = Dense(1024)(x)
    #x = LeakyReLU(0.1)(x)

    z_mean = Dense(nl, name='z_mean')(x)
    z_log_var = Dense(nl, name='z_log_var')(x)
    z = Lambda(sample_func, output_shape=(nl,), name='z')([z_mean, z_log_var])
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()

    latent_inputs = Input(shape=(nl,), name='z_sampling')

    #x = Dense(1024)(x)
    #x = LeakyReLU(0.1)(x)
    x = Dense(2048)(latent_inputs)
    x = LeakyReLU(0.1)(x)
    x = Dense(3136)(x)
    x = LeakyReLU(0.1)(x)
    x = Reshape((14, 14, 16))(x)
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    decoder = Model(latent_inputs, decoded)
    outputs = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img,outputs)
    #autoencoder.summary()

    #define custom loss function of the Beta-VAE
    def vae_loss(true, pred):
        rec_loss = mse(K.flatten(true), K.flatten(pred))
        rec_loss *= 224*224*3
        KL_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        KL_loss = K.sum(KL_loss, axis=-1)
        KL_loss *= -0.5
        vae_loss = K.mean(rec_loss + b*(KL_loss))
        return vae_loss

    #Define adam optimizer
    adam = keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer='adam',loss=vae_loss, metrics=[vae_loss])

    return autoencoder, encoder, z_log_var

#Train and save the B-VAE models
def train_w_s(X,autoencoder,epoch_number,batch_size_number,dir_path,save_path):
    X_train,X_test = X
    data = CSVLogger(dir_path + 'loss.csv', append=True, separator=';')
    filePath = save_path + 'weights.best.hdf5'#checkpoint weights
    checkpoint = ModelCheckpoint(filePath, monitor='vae_loss', verbose=1, save_best_only=True, mode='min')
    EarlyStopping(monitor='vae_loss', patience=20, verbose=0),
    callbacks_list = [checkpoint, data]
    autoencoder.fit(X_train, X_train,epochs=epoch_number,batch_size=batch_size_number,shuffle=True,validation_data=(X_test, X_test),callbacks=callbacks_list, verbose=2)

#Save the autoencoder model
def SaveAutoencoderModel(autoencoder,save_path):
	auto_model_json = autoencoder.to_json()
	with open(save_path + 'auto_model.json', "w") as json_file:
		json_file.write(auto_model_json)
	autoencoder.save_weights(save_path + 'auto_model.h5')
	print("Saved Autoencoder model to disk")

#Save the encoder model
def SaveEncoderModel(encoder,dir_path):
	en_model_json = encoder.to_json()
	with open(dir_path + '/en_model.json', "w") as json_file:
		json_file.write(en_model_json)
	encoder.save_weights(dir_path + '/en_model.h5')
	print("Saved Encoder model to disk")


#Test the trained models on a different test data
def test(autoencoder,encoder,test):
    autoencoder_res = autoencoder.predict(test)
    encoder_res = encoder.predict(test)
    res_x = test.copy()
    res_y = autoencoder_res.copy()
    res_x = res_x * 255
    res_y = res_y * 255

    return res_x, res_y, encoder_res

#Save the reconstructed test data in a separate folder.
#For this create a folder named results in the directory you are working in.
def savedata(test_in, test_out, test_encoded, Working_path, trainfolder):
    os.makedirs(trainfolder, exist_ok=True)
    for i in range(len(test_in)):
        test_in = np.reshape(test_in,[-1, 224,224,3])#Reshape the data
        test_out = np.reshape(test_out,[-1, 224,224,3])#Reshape the data
        cv2.imwrite(trainfolder + '/' + str(i) +'_in.png', test_in[i])
        cv2.imwrite(trainfolder + '/' + str(i) +'_out.png', test_out[i])


if __name__ == '__main__':
    train_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/data-generator/" #"/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    train_folders =  ["precipitation"]    #["clear-day","clear-day-50","evening","evening-50","mild-rain","mild-rain-50"]   #light - clear-day, clear-day-50      #rain - clear-day, mild-rain
    test_data_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    test_folders = ["heavy-rain-70"]
    dir_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/VAE/B-1.5/" #path to save the svdd weights
    save_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/VAE/B-1.5/" #path to save the svdd weights
    reconstruct_folder = dir_path + "reconstruction"
    #Loading images from the datasets
    csv_input = load_training_images(train_data_path, train_folders)
    csv_input = shuffle(csv_input)
    #Split the data to train and test. Then shuffle it for training
    X_train, X_test = np.array(csv_input[0:len(csv_input)-100].copy()), np.array(csv_input[len(csv_input)-100:len(csv_input)].copy())
    X_train = np.reshape(X_train, [-1, X_train.shape[1],X_train.shape[2],X_train.shape[3]])
    X_test = np.reshape(X_test, [-1, X_test.shape[1],X_test.shape[2],X_test.shape[3]])
    inp = (X_train,X_test)
    epoch_number=100 #epoch numbers for hyperparameter tuning and training
    batch_size_number=16 #batch size for hyperparameter tuning and training
    autoencoder,encoder,z_log_var = CreateModels(30,1.5)# Running the autoencoder model
    train_w_s(inp,autoencoder,epoch_number,batch_size_number,dir_path,save_path)#Train the selected B-VAE parameters and train the models and save them.
    SaveAutoencoderModel(autoencoder,save_path)#Save full autoencoder model
    SaveEncoderModel(autoencoder,save_path)#Save encoder model
    test_in, test_out, test_encoded = test(autoencoder,encoder,inp[1])#Test the autoencoder model with training weights.
    savedata(test_in, test_out, test_encoded, dir_path, reconstruct_folder)#Save the data
