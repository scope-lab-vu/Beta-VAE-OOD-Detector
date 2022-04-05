# Latent Space Encoding for B-VAE Hyperparameter Selection

Scripts here are used to select hyperparameters, the number of latent units (n), and Beta. We compared three hyperparameter tuning algorithms in this work, they are.

(1) Bayesian Optimization, (2) Random Search, and (3) Grid Search

To run the Bayesian optimization search using MIG as the optimization criteria, run the following. 

```
python3 bayes_mig.py              ---Bayesian optimization
```

To run other search algorithms using MIG as the optimization criteria, run the following.

```
python3 random_mig.py            ---random search

python3 grid_mig.py              ---grid search
```

To run other search algorithms using Elbo as the optimization criteria, run the following.

```
python3 bayes_elbo.py          ---bayesian optimization

python3 random_elbo.py        ---random search

python3 grid_elbo.py          ---grid search

```

**Note**: We require the images and CSV files generated in the [data generation](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/data-generation) step. mig_numgenerative, mig_sampling_value, and mig_iterations are parameters that need to be provided. **mig_numgenerative** is the number of labels (e.g., brightness, precipitation) that is known during data generation. **mig_sampling_value** is the number of sampling needed in computing MIG (default = 100). Sampling is required because MIG is based on entropy computation. Refer to the original paper on [MIG](https://proceedings.neurips.cc/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html) for more information. **mig_iterations** we compute an average MIG score. We do this to stabilize the score as it is based on sampling. 
