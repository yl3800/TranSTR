# [Raformer: Rationale-Empowered Transformer for Video Question Answering]()

In repo contains the code for "Raformer: Rationale-Empowered Transformer for Video Question Answering"


## Environment

Anaconda 4.10.3, python 3.7.13, pytorch 1.11.0 and cuda 11.3. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.7.13
>conda activate videoqa
>pip install -r requirements.txt 
```
## Data Preparation
Please download the pre-computed features and QA annotations from [MSRVTT , MSVD](https://drive.google.com/drive/folders/1JRPeEUW297xSY33Gf6z_Lx62ufgLLNO6?usp=sharing) , [NExT](https://github.com/doc-doc/NExT-QA) , [Causal-Vid](https://github.com/bcmi/Causal-VidQA).

After downloading the data, please put the data under the folder ```['video_feature']```  accordingly. Furthermore, you can modified the path in ['Dataloader.py'] to load the feature. 


## Usage
Once the data is ready, you can easily run the code. There are four folders whose names reprensent datasets. You can enter the folder accordingly. After entering a specific folder: 

First, to test the environment and code, we provide the prediction and weight of the models that indicate in the paper. You can get the results reported in the paper by running: 
>python train.py -v=test -m=test

The command above will load the best model file (click [here](https://drive.google.com/drive/folders/18uKR9LXhm4OHjVrNqUtPeOsDIqs5qkwz?usp=sharing) to download) under ['models/'], predict with it in test set and evaluate it. If you want to train the model, please run

>python train.py -v=train -m=train

It will train the model and save to ['models'].