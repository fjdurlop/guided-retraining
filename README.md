# Guiding the retraining of convolutional neural networks against adversarial inputs: best guidance metrics and configurations.

[![DOI](https://zenodo.org/badge/442607169.svg)](https://zenodo.org/badge/latestdoi/442607169)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Context: When using DL models, there are many possible vulnerabilities
and some of the most worrying are the adversarial inputs,
which can cause wrong decisions with minor perturbations.
Objective: We examined four guidance metrics for retraining
DL models and three retraining configurations, to improve them
against adversarials with regard to DL testing properties from the
point of view of a ml engineer in the context of DL models for
image classification.

Method: We conduced an empirical study in two datasets for
image classification.We explore: (a) the accuracy of ordering adversarial
inputs with four different guidance metrics (NC, DSA, LSA
and random), (b) the accuracy of retraining CNNs with three different
configurations (from scratch, using weights and the augmented
dataset, and using weights and only adversarial inputs).

Results: We reveal that retraining with adversarial inputs from
original model weights and by ordering with DSA gives the best
model w.r.t. accuracy and number of inputs used.

Conclusions: With the above configuration and metric, DL models
can improve against adversarial inputs without using many
inputs. We also show that dataset size has an important impact on
the results.


## Repository Structure

The repository is structured as follows:

<pre/>
- data
  | This folder contains the data after preprocessing, binary files in NumPy (.npy format)
- models
  | This folder contains our trained models
- notebooks
  | This folder contains the jupyter notebooks
- reports
  | Generated PDFs, graphics and figures to be used in reporting
- utils
  | Python functions
- requirements.txt: The dependencies of our implementation
</pre>


## Implementation of guided-retraining

### Datasets

<pre/>
- German Traffic Sign Recognition Benchmark (GTSRB)
  | URL: https://benchmark.ini.rub.de/
- Intel
  | URL: https://www.kaggle.com/puneet6060/intel-image-classification
</pre>

### GTSRB

- `notebooks/0.0.1-gtsrb-preprocessing.ipynb` - Preprocessing of original data.
- `notebooks/0.1.1-gtsrb-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.0.1-gtsrb-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.0.1-gtsrb-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA and random).
- `notebooks/2.0.2-gtsrb-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (NC).
- `notebooks/3-4.0.1-gtsrb-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.0.2-gtsrb-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2.
- `notebooks/3-4.0.3-gtsrb-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.

### Intel

- `notebooks/0.0.1-intel-preprocessing.ipynb` - Preprocessing of original data.
- `notebooks/0.1.1-intel-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.0.1-intel-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.0.1-intel-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA, NC and random).
- `notebooks/3-4.0.1-intel-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.0.2-intel-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2. 
- `notebooks/3-4.0.3-intel-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.


