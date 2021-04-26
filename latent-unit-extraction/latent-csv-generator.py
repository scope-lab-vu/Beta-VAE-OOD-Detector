#!/usr/bin/env python3
#script to take the trained B-VAE and the validation datasets to generate latent disctirbution csv.
#row[0] mean, row[1] logvar and row[2] samples.

#Libraries
import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from skimage import measure
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)

def results_transpose(res):
    ret = []
    for i in range(res.shape[1]):
        ret.append(res[:, i, :].copy())
    return np.array(ret)

#Test script which uses either encoder or autoencoder.
#Full:Use entire dataset at once for prediction, Partial:Use step prediction iterating through induvidual images.
#Autoencoder:Reconstructs the images, Encoder:Generates the latent space data.
def test(autoencoder,img,Faddress):
        csvfile = open(Faddress,'a+')
        writer = csv.DictWriter(csvfile,fieldnames=['mean', 'logvar', 'sample'])
        autoencoder_res = np.array(autoencoder.predict(img))
        autoencoder_res = results_transpose(autoencoder_res)
        print(len(autoencoder_res))
        for i in range(len(autoencoder_res)):
            auto={}
            auto['mean'] = autoencoder_res[i][0].tolist()
            auto['logvar'] = autoencoder_res[i][1].tolist()
            auto['sample'] = autoencoder_res[i][2].tolist()
            writer.writerow(auto)

#Load complete input images without shuffling
def load_images(folder_path,Folders):
    numImages = 0
    inputs = []
    path = folder_path + Folders + '/'
    print(path)
    numFiles = len(glob.glob1(path,'*.png'))
    numImages += numFiles
    print("Total number of images:%d" %(numImages))
    for img in glob.glob(path+'*.png'):
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        inputs.append(img)
    return inputs

#Function which reads a dataset and clusters it.
#For this create a unified dataset with training and testing dataset.
#This also writes to which cluster the latent data belons to.
def plotting(Working_path,dataset,latentsize,folder):
    train_distribution = []
    a = []
    m = []

    for i in range(2):
        train_distribution.append([])
        a.append([])

    for i in range (2):
        for j in range(latentsize):
            train_distribution[i].append([])

    for x in range(2):
        with open(dataset, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                  data = row[x].strip().split(',')
                  data[0] = data[0][1:]
                  data[len(data)-1]=data[len(data)-1][:-1]
                  data = np.array(data)
                  for y in range (latentsize):
                      train_distribution[x][y].append(float(data[y]))
    k=[]
    for z in range(len(train_distribution[0])):
            a[0].append(train_distribution[0][z])
            a[1].append(train_distribution[1][z])

            m=[]
            for x in range(len(train_distribution[0][0])):
                m.append(z)
            k.append(m)

    plt.scatter(a[0],a[1], c=k, s=20, cmap='viridis')
    plt.xlabel('Mean')
    plt.ylabel('LogVar')
    #plt.title('Z plots for ')
    #plt.range(-0.5,0.5)
    plt.colorbar()
    #path = Working_path + '/calibration-test'
    plt.savefig(Working_path + folder + '.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    #plt.show()


if __name__ == '__main__':
    #Load image dataset to be plotted
    path =  "/home/scope/Carla/CARLA_0.9.6/PythonAPI/new/dataset/"
    model_weights = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/B-VAE/B_1.5_L_30_ISORC/"
    Folders = ["clear-day","clear-day-50","clear-day-70"]
    x = 30
    for folder in Folders:
        image = load_images(path,folder)
        img = np.array(image[0:len(image)].copy())#test data
        img = np.reshape(img,[-1, 224,224,3])#Reshape the data
        File = folder + '.csv'
        Working_folder = "/home/scope/Carla/B-VAE-OOD-Monitor/latent-unit-extraction/results/"
        Fadress = Working_folder + File
        with open(model_weights + 'en_model.json', 'r') as jfile:
            autoencoder = model_from_json(jfile.read())
        autoencoder.load_weights(model_weights + 'en_model.h5')
        test(autoencoder,img,Fadress)#Run the test script
        plotting(Working_folder,Fadress,x,folder)
