#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main
import argparse

# name of weights store of main segmentation

m=main.main('proto')
m.train_proto('train')
m.train_proto('test')



