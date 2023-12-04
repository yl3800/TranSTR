# [TranSTR: Discovering Spatio-Temporal Rationales for Video Question Answering]()

In repo contains the code for "Discovering Spatio-Temporal Rationales for Video Question Answering"


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
Please download QA annotations from [NExT](https://github.com/doc-doc/NExT-QA) , [Causal-Vid](https://github.com/bcmi/Causal-VidQA).

After preparing the feature, please put the data under the folder ```['video_feature']```  accordingly. Furthermore, you can modified the path in ['Dataloader.py'] to load the feature. 


## Usage
Once the data is ready, you can easily run the code. There are four folders whose names reprensent datasets. You can enter the folder accordingly. After entering a specific folder: 

If you want to train the model, please run

>python train.py -v=train -m=train

It will train the model and save to ['models'].
