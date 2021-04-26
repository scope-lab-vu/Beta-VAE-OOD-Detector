#!/usr/bin/env python3
#Libraries
import os
import numpy as np
import csv
import statistics
seed = 7
np.random.seed(seed)
import prettytable
from prettytable import PrettyTable
from natsort import natsorted
from operator import itemgetter
from welford import Welford


def variance_calculator(path,top_n,w):
    variance = []
    data = []
    latent_unit = []
    with open(path,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            data.append(row)

    for i in range(len(data[0])):
        data1 = []
        latent_unit.append(i)
        for x in range(len(data)):
            data1.append(float(data[x][i]))
        variance_val = statistics.variance(data1)
        variance.append(round(variance_val,3))

    indices, variance_sorted = zip(*sorted(enumerate(variance), key=itemgetter(1),reverse=True))

    t = PrettyTable(['Latent Unit','kl-divergence difference'])
    for i in range(top_n):
        t.add_row([indices[i],variance_sorted[i]])
    print(t)


if __name__ == '__main__':
        w = Welford()
        top_n = 5
        path = "/home/scope/Carla/B-VAE-OOD-Monitor/latent-unit-extraction/results/"
        folders = ["clear-day-comparison.csv"]
        for folder in folders:
            data_path = path + folder
            variance_calculator(data_path,top_n,w)
