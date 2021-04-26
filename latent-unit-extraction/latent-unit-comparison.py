#!/usr/bin/env python3
#script takes in the latent units csv and computes the kl divergences from unit gaussian.
#For all the samples in the time series data, we find the average kl-divergence variance captured by that latent unit.
#If the latent unit captures only 1 generative factor and it shows a variance, it means the factor has changed.
#--------------selecting detector------------------------#
#Run this script across all the datasets with varying generative factors.
#example [low_illumination, medium_illumination, high_illumination, no_rain, low_rain, high_rain]
#For each of these datasets generate latent distributions using the trained B-VAE and latent_generator.py.
#Select union of latent units that show variance across all these runs.
#These latent unit set Ld = [L1,L2,l3,...,Ln] will be the detector latent units.
#--------------selecting diagnosers---------------------#
#Diagnoser latent unit(s) Lf will be a subset of Ld.
#We need to have varying levels of changes in the generative factors to identify diagnosers.
#example [low_illumination, medium_illumination, high_illumination]
# For these three sets, generate latent distributions using the trained B-VAE and latent_generator.py.
# Run the latent unit datasets using this script to generate the average latent unit kl-divergence variance.
# Compare the average latent unit variance across the three data sets.
# One or more latent units encoding information of illumination should show higher variance compared to others. Select it.

#Libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from math import log2
import math
from scipy.stats import norm
from statistics import mean
seed = 7
np.random.seed(seed)

def kl_divergence(Q, P):
     epsilon = 0.00001
     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence

    #Extracts the mean and logvar from the csv files.
    #computes the dissimilarity from unit gaussian using kl-divergence.
    #Averages it across the entire time-series data for each latent unit.
def KL_computer(Fadress,latentsize):
    train_distribution = []
    a = []
    m = []

    for i in range(2):
        train_distribution.append([])
        a.append([])

    for i in range(2):
        for j in range(latentsize):
            train_distribution[i].append([])


    for x in range(2):
        with open(Fadress, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                  data = row[x].strip().split(',')
                  data[0] = data[0][1:]
                  data[len(data)-1]=data[len(data)-1][:-1]
                  data = np.array(data)
                  for y in range (latentsize):
                      train_distribution[x][y].append(float(data[y]))
    kl=[]
    for z in range(latentsize):
        avg_klloss = []
        kl_val = 0.0
        for k in range(len(train_distribution[0][0])):
            mean = train_distribution[0][z][k]
            logvar = train_distribution[1][z][k]
            sd = math.sqrt(math.exp(logvar))
            x = np.arange(-10, 10, 0.001)
            p = norm.pdf(x, 0, 1)  # Normal Curve
            sum_p = np.sum(p)
            p[:] = [y / sum_p for y in p]
            q = norm.pdf(x, mean, sd) #
            sum_q = np.sum(q)
            q[:] = [z / sum_q for z in q]
            klloss = kl_divergence(q, p)

            avg_klloss.append(klloss)
        kl.append(avg_klloss)

    return kl

if __name__ == '__main__':
        latentsize=30
        monitor_list = []
        all_comparisons = []
        csv_list = ["clear-day","clear-day-50","clear-day-70"]
        path = "/home/scope/Carla/B-VAE-OOD-Monitor/latent-unit-extraction/results/"
        x = 0
        for elements in csv_list:
            print('-----------------------Run%d-----------------------'%x)
            csv_file = elements + '.csv'
            Faddress = path + csv_file
            kl_value = KL_computer(Faddress,latentsize)
            kl_comparison = []
            Higgins = []
            for j in range (0,len(kl_value[0]),2):
                Final_kl = []
                for k in range(30):
                    kl = abs(kl_value[k][j]- kl_value[k][j+1])
                    Final_kl.append(kl)
                kl_comparison.append(Final_kl)
            print(len(kl_comparison))
            for i in range(30):
                val = 0.0
                for k in range(len(kl_comparison)):
                    val += kl_comparison[k][i]
                val = val/len(kl_comparison)
                Higgins.append(val)
            print(Higgins)
            x+=1
            all_comparisons.append(Higgins)

        with open('/home/scope/Carla/B-VAE-OOD-Monitor/latent-unit-extraction/results/' + csv_list[0] + '-comparison.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_comparisons)
