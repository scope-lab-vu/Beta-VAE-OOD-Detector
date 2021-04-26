# Latent unit generation and extraction

These scripts generate latent distributions using a trained B-VAE. 
Plots the mean and logVar of the latent distirubtions and performs a latent comparison to find the set of detector latent units and diagnoser latent units.

```
run latent-csv-generator.ipynb        ---use the trained B-VAE weights to generate latent distribution csv's.

run latent-unit-comparison.ipynb      ---use generated latent distibutions to select detector latent units Ld and diagnoser latent units Lf 
```
