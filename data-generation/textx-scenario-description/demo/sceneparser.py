#!/usr/bin/python3
#input: Scene parameters for CARLA simulation
#Sun --> Value 90
#Cloudiness --> Range [0,50] sampler --> random
#Precipitation --> Range [0,50] sampler --> uniform
#Brightness --> Range [0,50] sampler --> uniform
#Road Segment --> Range [1,3] sampler --> random
#Output: Json file with sampled values for these simulation parameters

#Libraries
import textx
import numpy as np
import sys
import glob
import os
from textx.metamodel import metamodel_from_file
import csv
import json
import argparse
from argparse import RawTextHelpFormatter

#Reads the textX language and samples values for each scene parameter
def dump_scenario(scenarios):
    num_entities = len(scenarios.entities)
    params = {}
    for i in range(0,num_entities):
        entity_name = scenarios.entities[i].name
        num_parameters = len(scenarios.entities[i].properties)
        for j in range(0,num_parameters):
            data_type = scenarios.entities[i].properties[j].value.distribution
            if(data_type == 0):
                parameter_name = scenarios.entities[i].properties[j].name
                parameter_value = scenarios.entities[i].properties[j].value.min
                print(parameter_name, parameter_value)
            if(data_type == 1):
                parameter_name = scenarios.entities[i].properties[j].name
                parameter_min = scenarios.entities[i].properties[j].value.min
                parameter_max = scenarios.entities[i].properties[j].value.max
                sampler_type = scenarios.entities[i].properties[j].value.sampler
                if(sampler_type == "uniform"):
                    parameter_value = np.random.uniform(parameter_min,parameter_max)
                    parameter_value = int(parameter_value)
                elif(sampler_type == "normal"):
                    parameter_value = np.random.normal(parameter_min,parameter_max)
                    parameter_value = int(parameter_value)
                elif(sampler_type == "random"):
                    parameter_value = np.random.randint(parameter_min,parameter_max)
                print(parameter_name, parameter_min, parameter_max,sampler_type,parameter_value)
            params[parameter_name] = parameter_value

    return params


def main(arguments,store_path):
    scenario_meta = metamodel_from_file('entity.tx') #grammer for the scenario language
    for i in range(0,arguments.number):
        scenarios = scenario_meta.model_from_file('scene.entity') #scene entities
        os.makedirs(store_path + "Trial%d"%arguments.trial, exist_ok=True)
        out_file = open(store_path + "Trial%d"%arguments.trial + "/scene%d.json"%i, "w")
        params = dump_scenario(scenarios)
        json.dump(params, out_file, indent = 6)
        out_file.close()

if __name__ == '__main__':
        store_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-code/textx-scenario-description/scenes/"
        description = "Generate data for your Simulator\n"
        parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
        parser.add_argument('--simulator',
                        help='Choose from the simulator example 0 --> CARLA',type=int,required=False,default=0)
        parser.add_argument('--trial',
                        help='Choose the scenes for data generation',type=int,required=False,default=1)
        parser.add_argument('--number',
                        help='Enter the number of scenes you want to generate',
                        type=int,required=True,default=1)
        arguments = parser.parse_args()
        main(arguments,store_path)
