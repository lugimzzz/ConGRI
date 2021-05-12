# demo
    
# Description
ConGRI is a contrastive learning scheme based on deep Siamese CNN architecture, which automatically learns high-level feature embeddings for the expression images and then feeds the embeddings to an artificial neural network to determine whether or not the interaction exists.

# Dataset
To evaluate ConGRI, we use the eye dataset and mesoderm dataset.  ConGRI outperforms previous traditional and deep learning methods by a large margin, which achieves accuracies of 76.7% and 68.7% for the GRNs of early eye development and mesoderm development, respectively. 

# Model
we perform a two-stage learning procedure, corresponding to feature extraction and decision, respectively. In the first stage, we construct a Siamese network with two heads, whose inputs are the expression images of TF and candidate target gene, respectively. In the second stage,  we first average the extracted image features for each gene pair and then perform the prediction.
![image](https://user-images.githubusercontent.com/63761690/117983132-3e1da400-b369-11eb-822f-9f023ab56641.png)

Quik Start
=
Extracting the features by contrastive learning 
Feature extractor
-
Extracting the features by contrastive learning 

python train_valid.py

Decision module
-
Load and aggregate the feature embeddings extracted from the first stage

python aggregation.py

Predict the gene regulatory relationship in the gene-level by Multiple Instance Learning(MIL)

python decision_MIL.py
