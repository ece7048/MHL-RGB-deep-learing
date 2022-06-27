#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import patch_build

pb=patch_build.patch_build('train','covid_rgunet')
pb.patch_extract(class_list='off', patch_export='on',width=32,height=32, labels=[0,1,2,3,4,5,6,7,8,9], threshold=0.7)

