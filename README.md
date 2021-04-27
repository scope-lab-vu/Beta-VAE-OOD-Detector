# Efficient Out-of-Distribution Detection Using Latent Space of B-VAE for Cyber-Physical Systems

In this work we introduce the B-Variational Autoencoder (B-VAE) to detect out-of-Distribution images and identify the most likely feature (e.g.,brightness, precipitation,etc.) that caused the OOD. Typically, this is a multi-class classification problem solved using a chain of one-class classifiers. However, in this work we use a single efficient B-VAE deector that uses the principle of disentanglement to train the latent space to be sensitive to distribution shifts in different features. 

We demonstrate our approach using an end-to-end driving Autonomous Vehicle in the CARLA simulation. 

<p align="center">
  <img src="https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/videos/change-in-precipitation.gif" />
  <img src="https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/videos/change-in-brightness.gif" />
</p>

(a) HP scene where high precipitation is introduced into the scene. (b) HB scene where the brightness is introduced into the scenes. In HP scene the detector martingale and the martingale of the precipitation reasoner increases when the high precipitation is introduced. In the HB scene the detector martingale and the martingale of the brightness reasoner increases when the high precipitation is introduced. Additional videos are available [here]("https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/videos/")

# Setup

To run the CARLA setup with varied weather patterns and evaluate the B-VAE detector for OOD detection, clone this repo.

```
git clone https://github.com/scope-lab-vu/BVAE-OOD-Detector.git
```
Then create a conda environment with python 3.7 and install the requirements as follows.

```
To run this setup first create a virtual environment with python 3.7
conda create -n py37 python=3.7
conda activate py37
cd ${B-VAE-OOD-Monitor}  # Change ${B-VAE-OOD-Monitor} for your CARLA root folder
pip3 install -r requirements.txt
```
You will also need to install CARLA 0.9.6. See [link](https://carla.org/2019/07/12/release-0.9.6/) for more instructions.

# Steps to implement the B-VAE detector for OOD detection in CARLA simulator


1.  [Data generation](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/data-generation) -- Generate CARLA scenes with varied weather parameters. 

2. [DAVE-II-DNN-Training](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/lec-training) -- Train an NVIDIA DAVE-II LEC to steer the autonomous vehicle around Town1 of CARLA simulation. The LEC used here is based on [NVIDIA'S DAVE-II](https://arxiv-org.proxy.library.vanderbilt.edu/pdf/1604.07316.pdf?source=post_page---------------------------) DNN architecture.

2. [hyperparameter tuning](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/hyperparameter-tuning) -- Tune the B-VAE hyperparameters of Beta (B) and number of latent units (B). Three algorithms have been used in this work: Bayesian optimization, random search and grid search.

3. [BVAE_Train-Test](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/BVAE-Train-Test) -- Train the B-VAE with the selected hyperparameters (n,B) that was selected earlier.

3. [latent-unit-extraction](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/latent-unit-extraction) -- Extract and plot latent units using the selected B-VAE monitor. Then compare the latent variables across several scenes in a partition to select the detector latent variables (Ld) and reasoner latent variables (Lf)

4. [Carla-runtime-deployment](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/carla-runtime-deployment) -- Deploy the designed B-VAE detector in CARLA simulation to perform OOD detection.

# Earlier Work

1. Out-of-Distribution Detection in Multi-Label Datasets using Latent Space of β-VAE [paper](https://scopelab.ai/files/sundar2020detecting.pdf)

2. Efficient Multi-Class Out-of-Distribution Reasoning for Perception Based Networks: Work-in-Progress [paper](https://ieeexplore-ieee-org.proxy.library.vanderbilt.edu/stamp/stamp.jsp?tp=&arnumber=9244027)


# Extentions From the Earlier Work

1. Data generation -- Data generator using a Scenario Description Language (SDL) which uses random sampler to sample the CARLA weather parameters such as (sun, cloud, rain, illumination), and generate different scenes.

2. Bayesian Optimization Heuristic -- We perform Bayesian Optimization with [Mutual Information Gap](https://arxiv-org.proxy.library.vanderbilt.edu/pdf/1802.04942.pdf) as the optimization criteria to select the number of latent units (n) and B value. 

3. Latent Variable mapping Heuristic -- We perform a KL-divergence to perform a mapping between the latent variables and the sensitive features.

3. Comparison to existing technologies -- We compare our B-VAE detector with other techniques such as SVDD, VAE based reconstruction, SVDD-chain, and VAE-reconstruction-chain. 


