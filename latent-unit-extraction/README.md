# Latent Unit Mapping

Scripts to generate latent units and perform the latent unit mapping. The figure illustrates the latent unit mapping heuristic.

<p align="center">
   <img src="https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/blob/main/figures/latent-mapping.png" align="center" >
</p>

LP - Partition latent variables with high variance in average KL-divergence\
Ld - Detector latent units\
Lf - Reasoner Latent units

```

python3 latent-csv-generator.py    --use the trained B-VAE weights to generate CSV with latent unit parameters (mean, logvar, samples).

python3 latent-unit-comparison.py  --generate CSV with average KL-divergence of each latent unit for different scenes in a partition.

python3 latent-unit-selection.py   --uses Welford's variance calculator to return latent units Ld and Lf.

python3 latent-plotter.py      -- script to scatter plot induvidual latent unit. 

```

**Note**: These scripts will help you generate latent units using the trained B-VAE encoder. The latent unit comparison algorithm generates the KL divergence values of each latent unit using the partition information generated in the [data generation](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/data-generation) step. You will need to generate **train_kl.csv** and **calib_kl.csv** utilizing the training and calibration set generated in the data generation step. Next, use the **train_kl.csv** file with the latent-unit-selection script to identify the latent units for detection and factor identification.  
