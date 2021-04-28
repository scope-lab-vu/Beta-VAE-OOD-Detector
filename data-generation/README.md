# data generation
The script connects with CARLA 0.9.6 and generates different scenes by randomly varying CARLA weather parameters and road segments.

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
# Data generation using scene specification files

Once the scene specification json files are generated, they will be used by the simulator. sdl-data-generation.py script will read each scene specification file and uses it in the CARLA simulator. 

```
DISPLAY= ./CarlaUE4.sh -opengl    ---to start CARLA server (Terminal1)

./data_generation.sh              ---to start data_generator client (Terminal2)
```
Running the simulation generates images and labels are stored in a folder. A sample csv file with images and labels is available [here](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/data-generation/labels.csv)

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




