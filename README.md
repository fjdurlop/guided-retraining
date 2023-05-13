# 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7933484.svg)](https://doi.org/10.5281/zenodo.7933484)



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Background: When using deep learning models, one of the most critical vulnerabilities is their exposure to adversarial inputs, which can cause wrong decisions (e.g., incorrect classification of an image) with minor perturbations. To address this vulnerability, it becomes necessary to retrain the affected model against adversarial inputs, as part of the software testing process. In order to make this process energy efficient, data scientists need support on which are the best guidance metrics for reducing the adversarial inputs to use during testing, as well as optimal dataset configurations. 

Aim: We examined six guidance metrics for retraining deep learning models, specifically with convolutional neural network architecture, and three retraining configurations. Our goal is to improve the convolutional neural networks against the attack of adversarial inputs with regard to accuracy, resource utilization and execution time from the point of view of a data scientist in the context of image classification.

Method: We conduced an empirical study using four datasets for image classification. We explore: (a) the accuracy, resource utilization and execution time of retraining convolutional neural networks with the guidance of six different guidance metrics (neuron coverage, likelihood-based surprise adequacy, distance-based surprise adequacy, deepgini, softmax entropy and random), (b) the accuracy and resource utilization of retraining convolutional neural networks with three different configurations (from scratch and augmented dataset, using weights and augmented dataset, and using weights and only adversarial inputs).

Results: We reveal that retraining with an augmented training set with adversarial inputs, from original model weights, and by ordering with uncertainty metrics gives the best model w.r.t. accuracy, resource utilization and execution time.

Conclusions: Although more studies are necessary, we recommend data scientists to use the above configuration and metrics to deal with the vulnerability to adversarial inputs of deep learning models, as they can improve their models against adversarial inputs without using many inputs. We also show that dataset size has an important impact on the results.


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
- CIFAR-10
  | URL: http://www.cs.toronto.edu/~kriz/cifar.html
- MNIST
  | URL: http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST
  | URL: https://github.com/zalandoresearch/fashion-mnist

</pre>

### GTSRB

- `notebooks/0.0.1-gtsrb-preprocessing.ipynb` - Preprocessing of original data.
- `notebooks/0.1.1-gtsrb-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.1.1-gtsrb-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.1.1-gtsrb-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA and random).
- `notebooks/2.1.2-gtsrb-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (NC).
- `notebooks/3-4.1.1-gtsrb-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.1.2-gtsrb-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2.
- `notebooks/3-4.1.3-gtsrb-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.
- `notebooks/5.1.1-gtsrb-plot-c1.ipynb` - Plotting of configuration 1. 
- `notebooks/5.1.2-gtsrb-plot-c2.ipynb` - Plotting of configuration 2. 
- `notebooks/5.1.3-gtsrb-plot-c3.ipynb` - Plotting of configuration 3. 


### Intel

- `notebooks/0.0.1-intel-preprocessing.ipynb` - Preprocessing of original data.
- `notebooks/0.1.1-intel-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.1.1-intel-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.1.1-intel-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA, NC and random).
- `notebooks/3-4.1.1-intel-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.1.2-intel-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2. 
- `notebooks/3-4.1.3-intel-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.
- `notebooks/5.1.1-intel-plot-c1.ipynb` - Plotting of configuration 1. 
- `notebooks/5.1.2-intel-plot-c2.ipynb` - Plotting of configuration 2. 
- `notebooks/5.1.3-intel-plot-c3.ipynb` - Plotting of configuration 3. 

### CIFAR-10

- `notebooks/0.1.1-cifar-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.1.1-cifar-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.1.1-cifar-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA and random).
- `notebooks/2.1.2-cifar-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (NC).
- `notebooks/3-4.1.1-cifar-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.1.2-cifar-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2.
- `notebooks/3-4.1.3-cifar-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.
- `notebooks/5.1.1-cifar-plot-c1.ipynb` - Plotting of configuration 1. 
- `notebooks/5.1.2-cifar-plot-c2.ipynb` - Plotting of configuration 2. 
- `notebooks/5.1.3-cifar-plot-c3.ipynb` - Plotting of configuration 3. 

### MNIST

- `notebooks/0.1.1-mnist-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.1.1-mnist-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.1.1-mnist-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA and random).
- `notebooks/2.1.2-mnist-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (NC).
- `notebooks/3-4.1.1-mnist-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.1.2-mnist-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2.
- `notebooks/3-4.1.3-mnist-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.
- `notebooks/5.1.1-mnist-plot-c1.ipynb` - Plotting of configuration 1. 
- `notebooks/5.1.2-mnist-plot-c2.ipynb` - Plotting of configuration 2. 
- `notebooks/5.1.3-mnist-plot-c3.ipynb` - Plotting of configuration 3. 


### Fashion-MNIST

- `notebooks/0.1.1-fashion-original_model_training.ipynb` - Training of original model. 
- `notebooks/1.1.1-fashion-obtaining_adversarials.ipynb` - Step 1. Create adversarial inputs for training and testing.
- `notebooks/2.1.1-fashion-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (LSA, DSA and random).
- `notebooks/2.1.2-fashion-obtaining_metrics.ipynb` - Step 2. Obtain guidance metrics (NC).
- `notebooks/3-4.1.1-fashion-ordering_and_retraining-configuration_1.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 1.
- `notebooks/3-4.1.2-fashion-ordering_and_retraining-configuration_2.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 2.
- `notebooks/3-4.1.3-fashion-ordering_and_retraining-configuration_3.ipynb` - Step 3 and 4. Order inputs w.r.t. the guidance metrics and retraining according to configuration 3.
- `notebooks/5.1.1-fashion-plot-c1.ipynb` - Plotting of configuration 1. 
- `notebooks/5.1.2-fashion-plot-c2.ipynb` - Plotting of configuration 2. 
- `notebooks/5.1.3-fashion-plot-c3.ipynb` - Plotting of configuration 3. 
