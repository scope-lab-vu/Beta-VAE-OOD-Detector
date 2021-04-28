#!/usr/bin/env python2
import os
import sys
from itertools import product
import glob
import json
#from glob import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import argparse
import random
import time
import numpy as np
import math
import csv
import queue
from queue import Queue
import cv2
import scipy
import argparse
from argparse import RawTextHelpFormatter

IM_WIDTH = 640
IM_HEIGHT = 480

#Function to change weather setting in the CARLA simulator
def set_weather(c,p,s):
    weather = carla.WeatherParameters(
        cloudyness=c,
        precipitation=p,
        sun_altitude_angle=s)
    return weather

#Function to save steering actuation values.
def save_data(m,label1,label2,label3,data_folder):
    filename = data_folder+'/labels.csv'
    file_exists = os.path.isfile(filename)
    fields = ['frames',
                        'precipitation',
                        'brightness',
                        'road_segment']

    dict = [{'frames':'frame%d.png'%m,'precipitation':label1,'brightness':label2,'road_segment':label3}]

    with open(filename, 'a') as file:
            writer = csv.DictWriter(file, fieldnames = fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(dict)

#Function to process images and display a simulation window.
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    #return i3/255.0

#Function that artifically adds brightness to CARLA images.
#values for training: 30, 50. values for testing: 70
def increase_brightness(img, value):
    i1 = np.array(img.raw_data)
    i2 = i1.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    hsv = cv2.cvtColor(i3, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("", img)
    cv2.waitKey(1)
    return img

def generate_labels(sun,cloudiness,precipitation,bright,road):
    if(precipitation == 0):
        label1 = 0 #"no-rain"
    elif(precipitation >= 25):
        label1 = 1 #"low-rain"
    else:
        label1= 2 #"mild-rain"

    if(bright == 0):
        label2 =  3 #"not-bright"
    elif(bright >= 25):
        label2 =  4 #"low"
    else:
        label2= 5 #"mild-rain"

    if(road == 1):
        label3 = 6 #"road_segment=1"
    if(road == 2):
        label3 = 7 #"road_segment=3"
    if(road == 3):
        label3 = 8 #"road_segment=5"
    if(road == 6):
        label3 = 9 #"road_segment=6"
    else:
        label3 = 10
    return label1,label2,label3

#Main Function
def main(save_path,arguments,scene_list_data):
    m=0
    for i in range(len(scene_list_data)):
            world = None
            actor_list = []
            color_image_queue = queue.Queue()
            seg_image_queue = queue.Queue()
            col_queue = queue.Queue()
            label1,label2,label3 = generate_labels(scene_list_data[i]['sun'],scene_list_data[i]['cloudiness'],scene_list_data[i]['precipitation'],scene_list_data[i]['brightness'],scene_list_data[i]['road'])
            data_folder = save_path + "Trial%d"%(arguments.trial)
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            #os.makedirs(data_folder, exist_ok=True)
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)

            world = client.get_world()
            blueprint_library = world.get_blueprint_library()

            bp = random.choice(world.get_blueprint_library().filter("vehicle.bmw.grandtourer"))

            if bp.has_attribute('color'):
                color = (bp.get_attribute('color').recommended_values)[0]
                bp.set_attribute('color', color)

            transform = (world.get_map().get_spawn_points())
            spawn_point = transform[scene_list_data[i]['road']] #spawn point that can be used to change location of the vehicle in the simulator
            #So let's tell the world to spawn the vehicle.
            vehicle = world.spawn_actor(bp, spawn_point)

            actor_list.append(vehicle)
            print('created %s' % vehicle.type_id)

            settings = world.get_settings()
            settings.fixed_delta_seconds = 0.05 #FPS
            world.apply_settings(settings)

            #Color camera attached to the center of the Car's dashboard
            camera_bp = blueprint_library.find('sensor.camera.rgb') #depth
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '110')
            camera_transform = carla.Transform(carla.Location(x=10.5,y=0.0, z=0.7))
            center_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            actor_list.append(center_camera)

            colsensor_bp =blueprint_library.find('sensor.other.collision')#collision sensor value
            colsensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            actor_list.append(colsensor)

            print('created %s' % center_camera.type_id)
            time.sleep(4)
            center_camera.listen(color_image_queue.put)#center_camera image in a queue
            #Left_camera.listen(seg_image_queue.put)#Left_camera image in a queue
            colsensor.listen(col_queue.put)#collision sensor image in a queue
            weather = set_weather(scene_list_data[i]['sun'],scene_list_data[i]['cloudiness'],scene_list_data[i]['precipitation'])
            world.set_weather(weather)#Function to set weather
            l=0 #variable to track image count
            x=1
            while (l<arguments.images):#number of images to be collected
                try:
                    vehicle.set_autopilot(True)
                    image = color_image_queue.get()
                    process_img(image)
                    image.save_to_disk(data_folder+ '/frame%d'% (m))#save images
                    save_data(m,label1,label2,label3,data_folder)
                    #save_data(vehicle.get_control().steer)#save steer data
                    print(round(vehicle.get_control().steer,3))
                    l+=1
                    m+=1
                except KeyboardInterrupt:
                    sys.exit()

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')

    return data_folder

def postprocess(data_folder):
    print("*************Post Processing*****************")
    with open(data_folder+"/labels.csv", 'rt') as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          i=0
          for row in reader:
            if(int(row[2])>3):
                if(int(row[2])==4):
                    value=25
                if(int(row[2])==5):
                    value=50
                image = cv2.imread(data_folder + '/' + row[0])
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(data_folder+"/frame%d.png"%i,img)
                i+=1

def train_calibration_split(data_folder):
    csvfile = open(data_folder + '/calibration.csv','a')
    calib_writer = csv.writer(csvfile)
    csvfile1 = open(data_folder + '/train.csv','a')
    train_writer = csv.writer(csvfile1)
    i=0
    train=[]
    calib=[]
    with open(data_folder+"/labels.csv", 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if(i%4==0):
                    calib_writer.writerow(row)
                else:
                    train_writer.writerow(row)
                i+=1


if __name__ == '__main__':
    description = "Generate data for CARLA Simulator\n"
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('--images',
                    help='Choose number of images for each scene',type=int,required=False,default=250)
    parser.add_argument('--trial',
                    help='Choose the scenes for data generation',type=int,required=False,default=1)
    arguments = parser.parse_args()
    save_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/trial-data/" #path to save the generated data
    scene_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-code/textx-scenario-description/scenes/"
    scene_list_data  =[]
    scene_list = glob.glob(scene_path + "Trial%d"%arguments.trial + '/' + './*.json')
    for file in scene_list:
        f = open(file,)
        scene_data = json.load(f)
        scene_list_data.append(scene_data)
    data_folder=main(save_path,arguments,scene_list_data)
    postprocess(data_folder)
    train_calibration_split(data_folder)
