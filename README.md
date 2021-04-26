# Out-of-Distribution Detection and Reasoning using Latent Space of B-Variational Autoencoder

In this work we introduce the B-Variational Autoencoder (B-VAE) to detect out-of-Distribution images and identify the most likely feature (e.g.,brightness, precipitation,etc.) that caused the OOD. Typically, this is a multi-class classification problem solved using a chain of one-class classifiers. However, in this work we use a single efficient B-VAE deector that uses the principle of disentanglement to train the latent space to be sensitive to distribution shifts in different features. 

We demonstrate our approach using an end-to-end driving controller in the CARLA simulator. 

<p align="center">
  <img src="https://github.com/Shreyasramakrishna90/B-VAE-OOD-Monitor/blob/master/videos/change-in-brightness.gif" />
  <img src="https://github.com/Shreyasramakrishna90/B-VAE-OOD-Monitor/blob/master/videos/change-in-precipitation.gif" />
</p>

# Setup

To run the CARLA setup with varied weather patterns and evaluate the B-VAE monitor, clone this repo.

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

# Steps to implement the B-VAE monitor for OOD detection in CARLA simulator


1.  [Data generation](https://github.com/scope-lab-vu/BVAE-OOD-Detector/tree/main/data-generation) -- Generate CARLA scenes with varied weather parameters. 

2. [lec-training](https://github.com/scope-lab-vu/BVAE-OOD-Detector/tree/main/lec-training) -- Train an end-to-end LEC to steer the autonomous vehicle around CARLA towns. The LEC used here is based on [NVIDIA'S DAVE-II](https://arxiv-org.proxy.library.vanderbilt.edu/pdf/1604.07316.pdf?source=post_page---------------------------) DNN architecture.

2. [hyperparameter tuning](https://github.com/scope-lab-vu/BVAE-OOD-Detector/tree/main/hyperparameter-tuning) -- Tune the B-VAE hyperparameters of Beta (B) and number of latent units (B). Three algorithms have been used in this work: Bayesian optimization, random search and grid search.

3. [latent-unit-extraction](https://github.com/scope-lab-vu/BVAE-OOD-Detector/tree/main/latent-unit-extraction) -- Extract and plot latent units using the selected B-VAE monitor. Then compare the latent units across several CARLA scenes to select the detector latent units (Ld) and diagnoser latent units (Lf)

4. [Carla-runtime-deployment](https://github.com/scope-lab-vu/BVAE-OOD-Detector/tree/main/carla-runtime-deployment) -- Deploy the designed B-VAE monitor in CARLA simulation.

# Earlier Work

1. Out-of-Distribution Detection in Multi-Label Datasets using Latent Space of β-VAE [paper](https://scopelab.ai/files/sundar2020detecting.pdf)

2. Efficient Multi-Class Out-of-Distribution Reasoning for Perception Based Networks: Work-in-Progress [paper](https://ieeexplore-ieee-org.proxy.library.vanderbilt.edu/stamp/stamp.jsp?tp=&arnumber=9244027)


# Extentions From the Earlier Work

1. Data generation -- Data generator using a Scenario Description Language (SDL) which uses random sampler to sample the CARLA weather parameters such as (sun, cloud, rain, illumination), and generate different scenes.

2. Bayesian Optimization Heuristic -- We perform Bayesian Optimization with [Mutual Information Gap](https://arxiv-org.proxy.library.vanderbilt.edu/pdf/1802.04942.pdf) as the optimization criteria to select the number of latent units (n) and B value. 

3. Latent Variable mapping Heuristic -- We perform a KL-divergence to perform a mapping between the latent variables and the features.

3. Comparison to existing technologies -- We compare our B-VAE monitor (detection + diagnosis) with other techniques such as SVDD, VAE based reconstruction, SVDD-chain, and VAE-reconstruction-chain. 


