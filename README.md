# demo
    
# Description
ConGRI is a contrastive learning scheme based on deep Siamese CNN architecture, which automatically learns high-level feature embeddings for the expression images and then feeds the embeddings to an artificial neural network to determine whether or not the interaction exists.

# Dataset
To evaluate ConGRI, we use the eye dataset and mesoderm dataset.  ConGRI outperforms previous traditional and deep learning methods by a large margin, which achieves accuracies of 76.7% and 68.7% for the GRNs of early eye development and mesoderm development, respectively. 

# Code
we perform a two-stage learning procedure, corresponding to feature extraction and decision, respectively. In the first stage, we construct a Siamese network with two heads, whose inputs are the expression images of TF and candidate target gene, respectively. 
In the second stage,  we first average the extracted image features for each gene pair and then perform the prediction.

Quik Start
=

1.Feature extraction
-
python train_valid.py

2.Decision module
-
python decision.py
