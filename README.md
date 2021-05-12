ConGRI
=

Reverse engineering of gene regulatory networks (GRNs) has been an attractive research topic in system biology. In recent years, high-throughput spatial gene expression data has shed light on breakthrough development for the inference of GRNs. ConGRI is a contrastive learning scheme based on deep Siamese CNN architecture, which identifies gene regulatory interactions from gene expression images.

# Dataset
To evaluate ConGRI, we use the eye dataset and mesoderm dataset.  ConGRI outperforms previous traditional and deep learning methods by a large margin, which achieves accuracies of 76.7% and 68.7% for the GRNs of early eye development and mesoderm development, respectively. 

![image](https://user-images.githubusercontent.com/63761690/117985769-90f85b00-b36b-11eb-94ee-9334cdbf7cb4.png)


# Code
we perform a two-stage learning procedure, corresponding to feature extraction and decision, respectively. In the first stage, we construct a Siamese network with two heads, whose inputs are the expression images of TF and candidate target gene, respectively. In the second stage,  we first average the extracted image features for each gene pair and then perform the prediction.

![image](https://user-images.githubusercontent.com/63761690/117983132-3e1da400-b369-11eb-822f-9f023ab56641.png)


Feature extractor
-
Extracting the features by contrastive learning 

    python train_valid.py

Decision
-
Load and aggregate the feature embeddings extracted from the first stage

    python embedding_loading.py
    python aggregation.py

Predict the gene regulatory relationship in the gene-level by Multiple Instance Learning(MIL)

    python decision_MIL.py
