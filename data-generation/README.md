# Data generation
The script connects with CARLA 0.9.6 and generates different scenes by randomly varying CARLA weather parameters. To generate data run the following

```
DISPLAY= ./CarlaUE4.sh -opengl    ---to start CARLA server (Terminal1)

./data_generation.sh              ---to start data_generator client (Terminal2)
```

Launching data_generation.sh script will prompt a few simulation parameters. (1) the number of scenes to be generated, (2) the number of images for each scene, and (3) the folder to store the scenes


