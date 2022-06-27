#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn=main.main('binary_cancer_resnet',["1","2"])


# run train of main segmentation
mn.train_run(backbone_n="resnet50",name='binary_cancer')


