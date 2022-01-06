# Guiding the retraining of convolutional neural networks against adversarial inputs: best guidance metrics and configurations

[![DOI](https://zenodo.org/badge/442607169.svg)](https://zenodo.org/badge/latestdoi/442607169)

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

## Implementation of guided-retraining

## Repository Structure

The repository is structured as follows:

<pre/>
- models
  | This folder contains our trained models
- notebooks
  | This folder contains the jupyter notebooks
- reports
  | Generated PDFs, graphics and figures to be used in reporting
- requirements.txt: The dependencies of our implementation
</pre>