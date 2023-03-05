<h2 align="center">
Invariant Grounding for Video Question Answering 🔥
</h2>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/yl3800/IGV) 
[![](https://img.shields.io/badge/video-red?style=plastic&logo=airplayvideo)](https://youtu.be/wJhR9_dcsaM) 
</div>


## Overview 
This repo contains source code for **Invariant Grounding for Video Question Answering** (CVPR 2022 Oral, Best Paper Finalists). In this work, propose a new learning framework, Invariant Grounding for VideoQA (**IGV**), to ground the question-critical scene, whose causal relations with answers are invariant across different interventions on the complement. With IGV, the VideoQA models are forced to shield the answering process from the negative influence of spurious correlations, which significantly improves the reasoning ability.
    
<figure> <img src="figures/interventional-distributions.png" height="220"></figure>

## Installation
- Main packages: PyTorch = 1.11 
- See `requirements.txt` for other packages.

## Data Preparation
We use MSVD-QA as an example to help get farmiliar with the code. Please download the pre-computed features and trained models [here](https://drive.google.com/file/d/1MrupFq8jubEA4nEl4CppR5Rddz9rW_6Z/view?usp=sharing)

After downloading the data, please modify your data path in `run.py`.

## Run IGV

Simply run `run.sh` to reproduce the results in the paper. 


## Reference 
```
@InProceedings{Li_2022_CVPR,
    author    = {Li, Yicong and Wang, Xiang and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title     = {Invariant Grounding for Video Question Answering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2928-2937}
}
```