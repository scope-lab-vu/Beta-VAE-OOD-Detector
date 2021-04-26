#!/usr/bin/env python3
import glob
import os
import sys
from itertools import product

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

IM_WIDTH = 224
IM_HEIGHT = 224

#Function to change weather setting in the CARLA simulator
def set_weather(c,p,s):
    weather = carla.WeatherParameters(
        cloudyness=c,
        precipitation=p,
        sun_altitude_angle=s)
    return weather

#Function to save steering actuation values.
def save_data(m,label1,label2,label3,data_folder,steer):
    with open(data_folder+'/labels.csv', 'a') as file:
        val=[]
        writer = csv.writer(file)
        val.append('frame%d.png'%m)
        val.append(label1)
        val.append(label2)
        #val.append(p)
        val.append(label3)
        #val.append(bright)
        writer.writerow(val)

    with open(data_folder+'/steer.csv', 'a') as file:
        val1=[]
        writer = csv.writer(file)
        val1.append(steer)
        writer.writerow(val1)

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

def generate_labels(perception,sun,scene,bright):
    if(perception == 0):
        label1 = 0 #"no-rain"
    elif(perception == 25):
        label1 = 1 #"low-rain"
    else:
        label1= 2 #"mild-rain"

    if(bright == 0):
        label2 =  3 #"not-bright"
    elif(bright == 25):
        label2 =  4 #"low"
    else:
        label2= 5 #"mild-rain"

    if(scene == 1):
        label3 = 6 #"scene=1"
    if(scene == 3):
        label3 = 7 #"scene=3"
    if(scene == 5):
        label3 = 8 #"scene=5"
    if(scene == 6):
        label3 = 9 #"scene=6"
    return label1,label2,label3

#Main Function
def main(iterations,num_images,save_path,data_trial):
    c=[0.0] #0.0 = no cloud
    p=[0] #0 = no, 25=low, 50=mild, 70=high
    s=[100.0]#,90.0,80.0] #100, 90, 80 = day
    scene = [1,5]
    brightness = [0,25,50]#,25,50] #0=no, 25=low, 50=medium, 70=high
    m=0
    combinations_list = [list(x) for x in product(c,p,s,scene,brightness)]
    random_combinations_index = np.random.choice(range(0,len(combinations_list)), int(iterations), replace=False)
    combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]
    print(combinations_random_chosen)
    for i in range(int(iterations)):
            world = None
            actor_list = []
            color_image_queue = queue.Queue()
            seg_image_queue = queue.Queue()
            col_queue = queue.Queue()
            label1,label2,label3 = generate_labels(combinations_random_chosen[i][1],combinations_random_chosen[i][2],combinations_random_chosen[i][3],combinations_random_chosen[i][4])
            data_folder = save_path + "scene%d"%(data_trial)
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
            spawn_point = transform[combinations_random_chosen[i][3]] #spawn point that can be used to change location of the vehicle in the simulator
            #So let's tell the world to spawn the vehicle.
            vehicle = world.spawn_actor(bp, spawn_point)

            actor_list.append(vehicle)
            print('created %s' % vehicle.type_id)

            settings = world.get_settings()
            settings.fixed_delta_seconds = 0.05 #FPS
            world.apply_settings(settings)

            #Color camera attached to the center of the Car's dashboard
            camera_bp = blueprint_library.find('sensor.camera.rgb') #depth
            camera_bp.set_attribute('image_size_x', '224')
            camera_bp.set_attribute('image_size_y', '224')
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
            weather = set_weather(combinations_random_chosen[i][0],combinations_random_chosen[i][1],combinations_random_chosen[i][2])
            world.set_weather(weather)#Function to set weather
            l=0 #variable to track image count
            x=1
            while (l<num_images):#number of images to be collected
                try:
                    vehicle.set_autopilot(True)
                    image = color_image_queue.get()
                    process_img(image)
                    image.save_to_disk(data_folder+ '/frame%d'% (m))#save images
                    #save_data(vehicle.get_control().steer)#save steer data
                    print(round(vehicle.get_control().steer,3))
                    save_data(m,label1,label2,label3,data_folder,round(vehicle.get_control().steer,3))
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
    save_path = "/home/scope/Carla/B-VAE-OOD-Monitor/data-generation/results/" #path to save the svdd weights
    iterations = input("enter number of scene to be generated:")
    num_images = int(input("Enter the number of images for each scene:"))
    data_trial = int(input("Enter the folder to store the scenes:"))
    data_folder=main(iterations,num_images,save_path,data_trial)
    postprocess(data_folder)
    train_calibration_split(data_folder)
