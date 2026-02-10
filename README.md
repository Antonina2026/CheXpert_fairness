# Bias in Healthcare AI Algorithms: Detection and Mitigation

This repository contains the code and data for the master's thesis study.

## CheXpert dataset

The CheXpert dataset for chest X-ray radiograph interpretation can be downloaded [here](https://stanfordmlgroup.github.io/competitions/chexpert/#chexpert-dataset).  In this study, training data
from the VisualCheXbert labeler was used with positive (1) and negative (0) labels, file train_visualCheXbert.csv. The valid.csv file was used for validation.

## Experiment roadmap

In this experiment, all computations were processed using Python (versions 3.12.3 and 3.11.13), PyTorch (versions 2.9.1+cpu and 2.6.0+cu124), LibAUC (version 1.2.0), Geomloss (version 0.2.6), implemented in a Jupyter Notebook environment.

The Deep AUC model proposed by [Yuan et al. (2021)] (https://arxiv.org/abs/2012.03173) was selected as the baseline model, as it does not incorporate explicit fairness considerations. To enhance the fairness of the Deep AUC model, the Fair Identity Scaling (FIS) method proposed by [Luo et al. (2024](https://arxiv.org/abs/2310.02492) was employed.


