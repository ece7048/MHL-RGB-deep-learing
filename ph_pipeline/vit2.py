#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from covid_pipeline import main

# name of weights store of main segmentation
mn=main.main('vit')

# run train of main segmentation
mn.train_vit()



