#!/usr/bin/env python2
#bins:  0 no/negligible
       #1 mild
       #2 medium
       #3 high

import os
import sys
from itertools import product
import glob
import json
import random
import time
import numpy as np
import math
import csv
import cv2
import scipy

#Main Function
def read_data(dataset_path):
    images = []
    labels = []
    opened_csv_file = open(dataset_path + 'labels.csv', 'r')
    reader = csv.reader(opened_csv_file)
    next(reader)
    for row in reader:
        label_val = []
        images.append(row[0])
        label_val.append(int(row[1]))
        label_val.append(int(row[2]))
        label_val.append(int(row[3]))
        label_val.append(int(row[4]))
        label_val.append(int(row[5]))
        labels.append(label_val)

    return images,labels

def label_value_normalization(labels):
    j = 0
    normalized_labels = []
    for j in range(len(labels[0])):
        my_labels = []
        normalize = []
        for i in range(len(labels)):
            my_labels.append(int(labels[i][j]))

        for i in range(len(my_labels)):
            #print((my_labels[i] - min(my_labels))/(max(my_labels)-min(my_labels)))
            normalize.append((my_labels[i] - min(my_labels))/(max(my_labels)-min(my_labels)))
        #print(normalize)
        normalized_labels.append(normalize)

    return normalized_labels


def label_binning(labels):
    binned_labels = []
    for i in range(len(labels[0])):
        #print(j)
        bins = []
        for j in range(len(labels)):
            #print(labels[i][j])
            if(labels[j][i]>=0 and labels[j][i]<0.25):
                bins.append(0)
            elif(labels[j][i]>=0.25 and labels[j][i]<0.5):
                bins.append(1)
            elif(labels[j][i]>=0.5 and labels[j][i]<0.75):
                bins.append(2)
            elif(labels[j][i]>=0.75 and labels[j][i]<=1.0):
                bins.append(3)
        #print(bins)
            # elif(labels[i][j]>=0.8 and labels[i][j]<=1.0):
            #     bins.append(4)
        binned_labels.append(bins)

    return binned_labels

def partition(images,labels):
    partition_brightness = []
    partition_precipitation = []
    no_change_partition = []
    both_change_partition = []
    for i in range(len(images)):
        if(labels[i][2]==0 and labels[i][3]==0):
            no_change_partition.append(images[i])
        elif(labels[i][2]!=0 and labels[i][3]==0 and (labels[i][1]<=2)): #or labels[i][1]!='5')):
            partition_precipitation.append(images[i])
        elif(labels[i][2]==0 and labels[i][3]!=0 and (labels[i][1]<=2)): #or labels[i][0]!='2')):
            partition_brightness.append(images[i])
        elif(labels[i][2]!=0 and labels[i][3]!=0):
            both_change_partition.append(images[i])

    print(partition_brightness)

    print("Length of no change partition:%d"%len(no_change_partition))
    print("Length of precipitation partition:%d"%len(partition_precipitation))
    print("Length of brightness partition:%d"%len(partition_brightness))
    print("Length of both change partition:%d"%len(both_change_partition))



if __name__ == '__main__':
    dataset_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/trial-data/Trial2/" #path to save the generated data
    image, labels = read_data(dataset_path)
    normalized_labels = label_value_normalization(labels)
    binned_labels = label_binning(normalized_labels)
    partition(image,binned_labels)
