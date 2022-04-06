# System Architecture

The component architecture implemented in CARLA is:

<p align="center">
   <img src="https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/figures/Carla-block-diagram.png" align="center" >
</p>

# Testing B-VAE monitor in CARLA

```
./launch.sh                     ---to use the trained B-VAE monitor in the CARLA simulator

```
**Note**: detector.py and detector1.py scripts perform detection and feature identification. You will need to add the detector latent units and the reasoner latent units we identified in the previous step. Please enter these latent units as class_detector and class_detector2. You will also need to have the calibration.csv file initially generated in the data generation step. 
