#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation


mn=main.main('')
model_path= '../../python_source/Model_covid/weights_denseres171_denseres171.h5'                                                                          
model_path1= '../../python_source/Model_covid/weights_resnet50_resnet50.h5'
model_path2= '../../python_source/Model_covid/weights_densenet121_densenet121.h5'
img_path='/../../fastdata/mer17mm/private/Data/cov_4_2/'

layer="conv5_block16_concat"
output_path1='/../../fastdata/mer17mm/private/Data/map_attention/dense/'

layer2="res5c_branch2a"
output_path2='/../../fastdata/mer17mm/private/Data/map_attention/res/'

layer3='concatenatef3'
output_path3='/../../fastdata/mer17mm/private/Data/map_attention/dr/'

layerall=[layer,layer2,layer3]
output_pathall=[output_path1,output_path2,output_path3]
model=[model_path2, model_path1, model_path]
m_n=['densenet121', 'resnet50', 'denseres171']
mn.attention_map(img_path, output_pathall, m_n, model, layerall)
