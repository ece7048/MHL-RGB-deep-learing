#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn=main.main('patch_ph32',["normal","GG","emphysema"])
image= "../../../DATA/nii_patches/miccai/four/normal/Image_210_1.jpeg"
model="../ph_pipeline/Model/weights_miccai_200ph_vgg16.h5"
store= "../../../DATA/nii_patches/miccai/four/XAI/GG/imageN.png"
point=[1,11]
# run train of main segmentation
mn.xai(image,model,store,point,model_name='vgg16',height=224,width=224,channels=1,classes=4,case='test',batch=1,label=[1,0,0,0],rot=180)

#me of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn1=main.main('patch_ph32',["normal","GG","emphysema"])
image= "../../../DATA/nii_patches/miccai/four/GG/Image_220_3.jpeg"
model="../ph_pipeline/Model/weights_miccai_200ph_vgg16.h5"
store= "../../../DATA/nii_patches/miccai/four/XAI/GG/imageG.png"
point=[1,11]
# run train of main segmentation
mn1.xai(image,model,store,point,model_name='vgg16',height=224,width=224,channels=1,classes=4,case='test',batch=1,label=[0,1,0,0],rot=180)

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn2=main.main('patch_ph32',["normal","GG","emphysema"])
image= "../../../DATA/nii_patches/miccai/four/honeycomb/Image_185_6.jpeg"
model="../ph_pipeline/Model/weights_miccai_200ph_vgg16.h5"
store= "../../../DATA/nii_patches/miccai/four/XAI/GG/imageH.png"
point=[1,11]
# run train of main segmentation
mn2.xai(image,model,store,point,model_name='vgg16',height=224,width=224,channels=1,classes=4,case='test',batch=1,label=[0,0,1,0],rot=180)

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn3=main.main('patch_ph32',["normal","GG","emphysema"])
image= "../../../DATA/nii_patches/miccai/four/emphysema/Image_100_1_9.jpeg"
model="../ph_pipeline/Model/weights_miccai_200ph_vgg16.h5"
store= "../../../DATA/nii_patches/miccai/four/XAI/GG/imageE.png"
point=[1,11]
# run train of main segmentation
mn3.xai(image,model,store,point,model_name='vgg16',height=224,width=224,channels=1,classes=4,case='test',batch=1,label=[0,0,0,1],rot=180)

