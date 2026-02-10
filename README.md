# Bias in Healthcare AI Algorithms: Detection and Mitigation

This repository contains the code and data for the master's thesis study.

## CheXpert dataset

The CheXpert dataset for chest X-ray radiograph interpretation can be downloaded [here](https://stanfordmlgroup.github.io/competitions/chexpert/#chexpert-dataset).  In this study, training data
from the VisualCheXbert labeler was used with positive (1) and negative (0) labels, file train_visualCheXbert.csv. The valid.csv file was used for validation.

## Experiment roadmap

In this experiment, all computations were processed using Python (versions 3.12.3 and 3.11.13), PyTorch (versions 2.9.1+cpu and 2.6.0+cu124), LibAUC (version 1.2.0), Geomloss (version 0.2.6), implemented in a Jupyter Notebook environment.

The Deep AUC model proposed by [Yuan et al. (2021)](https://arxiv.org/abs/2012.03173) was selected as the baseline model, as it does not incorporate explicit fairness considerations. To enhance the fairness of the Deep AUC model, the Fair Identity Scaling (FIS) method proposed by [Luo et al. (2024)](https://arxiv.org/abs/2310.02492) was employed.

### Pre-processing steps

All pre-processing steps were performed on a standard laptop with Windows 11 (version 10.0.26100, SP0) operating system. The data analysis was performed using script [CheXpert_preprocessing.ipynb](https://github.com/Antonina2026/CheXpert_fairness/blob/main/code/CheXpert_preprocessing.ipynb) and the CheXpert files train_visualCheXbert.csv, valid.csv (placed in CheXpert-v1.0 zip file in folder [data](https://github.com/Antonina2026/CheXpert_fairness/tree/main/data). Also this script uses [train.csv](https://github.com/Antonina2026/CheXpert_fairness/blob/main/results/train.csv) file with the reduced number of samples. The reduced version of the original CheXpert dataset, as a result of this script, is placed on the Kaggle platform [CheXpert-v1.0](https://www.kaggle.com/datasets/antoninab/chexpert-v1-0/data). The training set was reduced to 155,122 radiographs. To mitigate sex imbalance, the male subgroup was undersampled via random sampling, enforcing a 50/50 male–female split in the total number of training samples and an approximately balanced distribution within each pathology. All training images were resized to 512×512 pixels using the Lanczos resampling filter. The validation set remained unchanged, consisting of 234 radiographs.

### Model training

All three models were trained on the Kaggle platform using script [CheXpert_training_deepauc_fis.ipynb](https://github.com/Antonina2026/CheXpert_fairness/blob/main/code/CheXpert_training_deepauc_fis.ipynb).

### Model fairness assessment

The model fairness assessment was performed on a standard laptop with Windows 11 (version 10.0.26100, SP0) operating system. The data analysis was performed using script [CheXpert_validation.ipynb](https://github.com/Antonina2026/CheXpert_fairness/blob/main/code/CheXpert_validation.ipynb). This scrip uses three .csv files that were created during the training process. The pred_results_Deep_AUC.csv file contains predicted probabilities for the Deep AUC model, it does not incorporate explicit fairness considerations. The pred_results_FIS_sex.csv file contains predicted probabilities for the Deep AUC + Sex model, it incorporates FIS method for sex groups. The pred_results_FIS_age.csv file contains predicted probabilities for the Deep AUC + Age model, it incorporates FIS method for age groups. All three files were saved in folder [results](https://github.com/Antonina2026/CheXpert_fairness/tree/main/results).
