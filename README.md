# DPR-FR
Diffusion Purification for Robust Face Recognition

## Env
On your remote server (with GPUs available):
```
conda env create -f environment.yml
conda activate dpr-fr
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
If you add a package to the env file and want to update the conda env do:
``
conda env update -f environment.yml --prune
``

## Dataset
LFW: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download


## Baselines
To run the FR baseline:
``
python3 src/run_baseline_fr.py
``

## TODO:
1) Add a run_lfw_verification.py file that use the standard metadata files of the LFW and use the train-test splits to produce final metrics
