#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn=main.main('ph_200',['abnormal','combined','emphysema','fibrosis'])


# run train of main segmentation
mn.train_run()


