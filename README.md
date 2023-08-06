# Repository for Fully Bayesian ComBat

This is the home of Fully Bayesian Combat, an extension of [Longitudinal ComBat](https://github.com/jcbeer/longCombat) which uses fully Bayesian inference rather than empirical Bayes.

See the corresponding paper [here](https://doi.org/10.1016/j.nicl.2023.103472).

# What is BayesComBat?
BayesComBat is a fully Bayesian implementation of ComBat, a model often used for harmonization of brain imaging features (e.g. cortical thickness) from across multiple sites. ComBat models and removes scanner effects from the data.  


# Installation

Install from github:
```bash
git clone https://github.com/batmanlab/BayesComBat
```
or with pip:
```bash
pip install BayesComBat
```



# Usage

There are three functions used to harmonize data with BayesComBat. First, `infer` performs the Bayesian inference and saves MCMC objects in outdir. This is the longest step. The inference+save samples should be divided into iterations using `num_iterations` to prevent memory errors from saving large sample files.

```python
from BayesComBat import hamonize
import pandas as pd

# Load data
data = pd.read_csv('my_imaging_data.csv')

#specify features in data to harmonize 
features = ['feature1', 'feature2', 'feature3']

#specify biological covariates
covariates = ['covaritate1', 'covariate2']

harmonize.infer(data=data,
features=['feature1', 'feature2', 'feature3'],
covariates=['covaritate1', 'covariate2'],
batch_var='Scanner',
subject_var='Subj_ID',
outdir='/path/to/output/directory',
num_warmup=1000,
num_samples_per_iteration=1000,
num_iterations=10,
num_chains=4
)
```

Next, `harmonize` uses the saved MCMC samples to harmonize the data.

```python 
harmonize.harmonize(data=data,
features=['feature1', 'feature2', 'feature3'],
covariates=['covaritate1', 'covariate2'],
batch_var='Scanner',
subject_var='Subj_ID',
outdir='/path/to/output/directory',
num_iterations=10)
```

Finally, `load_harmonized_data` will load an array of size (num_samples,num_images,num_features) with the harmonized data. Note that num_sampples is num_chains*num_iterations*num_samples_per_iteration.

```python
harmonized_data = harmonize.load_harmonized_data(
    dir='/path/to/output/directory',
    num_iterations=10)
```

Note: for GPU-accelerated inference, Jax GPU Version should be installed separately (See [here](https://github.com/google/jax#installation)).





