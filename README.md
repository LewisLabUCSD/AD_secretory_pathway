# Expression-guided random walk 

This is the companion Github repo to the paper entitled "Dysregulation of the secretory pathway connects Alzheimer’s disease genetics to aggregate formation". 
The repo contains the code to calculate the secretory pathway support component and summary support scores for any secreted protein of interest. 
The scores quantify a cell's/ tissue's fitness for producing specific secreted proteins. 
We implemented an expression-guided random walk, which integrates protein-protein interaction (PPI) networks and cell-/ tissue-specific 
gene expression profiles to leverage the individuality of the transcriptome data. Implementation details can be found here: https://doi.org/10.1101/2020.08.10.243634 


## Getting started

An interactive notebook fully loaded with the required packages can be found here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LewisLabUCSD/AD_secretory_pathway/blob/master/Support_score_calculation_example.ipynb)

Alternatively, you can set up your own environment according to ```requirements.txt```. Note that for GPU-accelerated support score and graph gradient calculation, please install Pytorch with CUDA.

