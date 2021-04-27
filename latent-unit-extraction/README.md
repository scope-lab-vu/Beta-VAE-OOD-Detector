# Latent Variable Mapping

Scripts to generate latent variables, and perform latent variable mapping. 

<p align="center">
   <img src="https://github.com/Shreyasramakrishna90/OOD-B-VAE/blob/master/videos/block-carla.png" align="center" >
</p>

```

python3 latent-csv-generator.py    --use the trained B-VAE weights to generate csv with latent variable parameters (mean, logvar, samples).

python3 latent-unit-comparison.py  --generate csv with average kl-divergence of each latent variables for different scenes in a partition.

python3 latent-unit-selection.py   --uses Welford's variance calculator to return latent variables Ld and Lf.

python3 latent-plotter.py      -- script to scatter plot induvidual latent variables. 

```
