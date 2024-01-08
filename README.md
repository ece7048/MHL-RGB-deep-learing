# MHL-RGB-deep-learing

Multi-head attention channel

Metastases are the main cause of cancer mortality. Metastasis is caused by the defective cell migration of cancer cells and is accompanied by major changes in the spatial organisation of the cytoskeleton in cells, including the actin microfilaments and the vimentin intermediate filaments. Understanding of how these filaments change during the transformation from normal to invasive cells can provide molecular targets and strategies
for cancer diagnosis and therapy. 

We have used a high-resolution dataset of metastasising and normal cells. We test a classification between normal: Bj primary fibroblast cells, and their isogenically matched, transformed and invasive counterpart: BjTertSV40TRasV12. In this work we developed established deep learning networks and multi-attention channel architecture. To increase the interpretability of the network's black box, we developed explainable techniques like GradCam. To generalise and compare the GradCam results of the networks we developed a method using the weighted geometric mean of the total shape of the cell and their GradCam images. 

The source code is on the ph_pipeline/ repository
The test/ repository includes the run script (.py) and the configuration files (.config) we used to train all our models.

# Install python code

cd unzip_fold_of_code/

pip install .

# Source code description

In ph_pipeline folder the core code of the library is located.
The MHL and RGB family models are developed in the create_net.py file
The XAI/GRADCAM folder has all the explanable code of the publish paper [1].

Please if you use the code cite the above publication:
[1] https://arxiv.org/abs/2309.00911


[1]

 
