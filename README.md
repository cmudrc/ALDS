# ALDS
Adaptive Local Data Scoping for neural network learning of large-scale partial differential equations

## Introduction
This repository contains the code for the paper "". 

## Quick Start
To configure the environment, run the following command:
```bash
pip install -r requirements.txt
```

Training and prediction on ALDS, naive Data Scoping (DS), and full data (one-shot) can be done by running the three separate python scripts with appropriate configurations. One example experiment doing prediction on the John Hopkins Turbulence Database (JHTDB) with ALDS, with pre-trained PCA encoder, K-Means clustering, and Fourier Neural Operator (FNO) model, can be done by running the following command:
```bash
python3 run_ALDS.py --dataset=jhtdb --encoder=pca --classifier=kmeans --model=fno --exp_name=fno_jhtdb_alds --mode=pred --exp_config configs/exp_config/fno_jhtdb.yaml --train_config configs/train_config/fno.yaml 
```
After executing the command, the prediction results, figures as well as raw data will be saved in the folder `./logs/`.
