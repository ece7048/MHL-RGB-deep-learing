#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
#mn=main.main('binary_cancer',["normal","cancer"])
#image= "../../../DATA/cancer/1/BJprimary_c1-2.tif"
#model="../ph_pipeline/Model/weights_binary_cancer_RGB.h5"
#store= "../../../DATA/cancer/xai/healthy_rgb.png"
#point=[1,10]
# run train of main segmentation
#mn.xai(image,model,store,point,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180)

#mn1=main.main('binary_cancer',["normal","cancer"])
#image1= "../../../DATA/cancer/2/Copy of BJRas1_c1_50.tif"
#model1="../ph_pipeline/Model/weights_binary_cancer_RGB.h5"
#store1= "../../../DATA/cancer/xai/cancer_rgb.png"
#point1=[1,10]
# run train of main segmentation
#mn1.xai(image1,model1,store1,point1,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180)

#mne=main.main('binary_cancer',["normal","cancer"])
#imagee= "../../../DATA/cancer/1/BJprimary_c1-2.tif"
#modele="../ph_pipeline/Model/weights_binary_cancer_resnet_RGB.h5"
#storee= "../../../DATA/cancer/xai/healthy_rgb.png"
#pointe=[1,10]
# run train of main segmentation
#mne.xai(imagee,modele,storee,pointe,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180,backbone="resnet50")

#mn1w=main.main('binary_cancer',["normal","cancer"])
#image1w= "../../../DATA/cancer/2/Copy of BJRas1_c1_50.tif"
#model1w="../ph_pipeline/Model/weights_binary_cancer_resnet_RGB.h5"
#store1w= "../../../DATA/cancer/xai/cancer_rgb.png"
#point1w=[1,10]
# run train of main segmentation
#mn1w.xai(image1w,model1w,store1w,point1w,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180,backbone="resnet50")

#mneq=main.main('binary_cancer',["normal","cancer"])
#imageeq= "../../../DATA/cancer/1/BJprimary_c1-2.tif"
#modeleq="../ph_pipeline/Model/weights_binary_cancer_densenet_RGB.h5"
#storeeq= "../../../DATA/cancer/xai/healthy_rgb.png"
#pointeq=[1,10]
# run train of main segmentation
#mneq.xai(imageeq,modeleq,storeeq,pointeq,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180,backbone="densenet121")

#mn1wq=main.main('binary_cancer',["normal","cancer"])
#image1wq= "../../../DATA/cancer/2/Copy of BJRas1_c1_50.tif"
#model1wq="../ph_pipeline/Model/weights_binary_cancer_densenet_RGB.h5"
#store1wq= "../../../DATA/cancer/xai/cancer_rgb.png"
#point1wq=[1,10]
# run train of main segmentation
#mn1wq.xai(image1wq,model1wq,store1wq,point1wq,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180,backbone="densenet121")


# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn3=main.main('binary_cancer',["normal","cancer"])
image3= "../../../DATA/cancer/1/BJprimary_c1-2.tif"
model3="../ph_pipeline/Model/weights_binary_cancer_DENRES.h5"
store3= "../../../DATA/cancer/xai/healthy_deresn.png"
point3=[1,26]
# run train of main segmentation
mn3.xai(image3,model3,store3,point3,model_name='DENRES',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180)

mn2=main.main('binary_cancer',["normal","cancer"])
image12= "../../../DATA/cancer/2/Copy of BJRas1_c1_50.tif"
model12="../ph_pipeline/Model/weights_binary_cancer_DENRES.h5"
store12= "../../../DATA/cancer/xai/cancer_denres.png"
point12=[1,26]
# run train of main segmentation
mn2.xai(image12,model12,store12,point12,model_name='DENRES',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180)

# name of weights store of main segment
