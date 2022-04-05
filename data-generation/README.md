# Data Generation
The script connects with CARLA 0.9.6 and generates different scenes by randomly varying CARLA weather parameters and road segments. The outputs of this step are forward facing camera images and  

# Downloads
We used CARLA version 0.9.6 in the paper. You can get the pre-compiled simulator from [here](https://carla.readthedocs.io/en/0.9.6/download/). 

# TextX Based Scene Generation in CARLA Simulation

We use a scenario description DSML written in [textX](https://textx.github.io/textX/stable/) to generate different scenes with different weather patterns. 

[Scene.entity](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/demo/scene.entity) -- Has the entities of a CARLA scene. The entities are sun_angle, cloudiness, precipitation, brightness, and road_segments. These parameters can take a continuous or discrete value. 

[entity.tx](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/demo/entity.tx) -- Has the grammer for the scenario description language. 

[sceneparser.py](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/demo/sceneparser.py) -- Parses the scenario language as python object and fills in the required values for each scene parameter. Then a json file with the scene paramters is generated, which will be used by the simulator.

[scenes](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/scenes/) -- Has a sample scene specification json file for CARLA simulation. 

Scenario.dot, entity.dot -- metamodel figures of the scenario description and the textual language. Read the [docs](https://textx.github.io/textX/stable/) to convert it to png.

To generate scenes with different weather patterns, activate the virtual environment using source demo/bin/activate. Then run the following commands to generate different simulation scenarios.

```
cd demo
textx generate entity.tx --target dot
textx generate scene.entity --grammar entity.tx --target dot
python3 sceneparser.py 
```
The sceneparser script generates json files with values for different weather parameters and road semgnents as shown [here](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/scenes/Trial1/scene0.json). In the script you can select the number of scenes that you want to generate. Based on this, several json files will be generated in a sequential ordering as shown in this [folder](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/textx-scenario-description/scenes/Trial1). Next, these json files are read and loaded into the simulator to generate different scenes in the simulator. 

# Data generation using scene specification files

Once the scene specification json files are generated, they will be used by the simulator. sdl-data-generation.py script will read each scene specification file and uses it in the CARLA simulator. 

```
DISPLAY= ./CarlaUE4.sh -opengl    ---to start CARLA server (Terminal1)

./data_generation.sh              ---to start data_generator client (Terminal2)
```
The data_generation script reads the json files previously generated and then loads it into the simulator. 

**Output**: Running the data generation script with the simulator generates images and labels that are stored in a folder specified in the script. **The CSV file has lables of different environmental conditions (e.g., brightness, precipitation) and steering values**. A sample csv file with images and labels is available [here](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/labels.csv). The CSV file is also split into **train.csv** and **calibration.csv** files. The train.csv files is used to train the detector and the calibration file is used in the Inductive Conformal Prediction (ICP) procedure during runtime detection. 

**label Description**
label 0  - precipitation = 0%,
label 1  - precipitation <= 25%,
label 2  - precipitation > 25%,
label 3  - brightness = 0%,
label 4  - brightness <= 25%,
label 5  - brightness > 25%,
label 6  - road_segment = 0,
label 7  - road_segment = 1,
label 8  - road_segment = 2,
label 9  - road_segment = 6,
label 10  - road_segment = others,

# Data Partitioning

Once the label csv is generated, we can bin the data into several partitions based on the variance in label values. To generate partitions run the following script. These partitions are required for hyperparameter tuning and latent unit comparison. 

```
python2 data-bins-partitions.py
```





