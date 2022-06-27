#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import main

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
mn=main.main('mhl_cancer',["normal","cancer"])
image= "../../../DATA/cancer/1/"       #Copy of BJprimary2_9.tif"
model="../ph_pipeline/Model/weights_binary_cancer_densenet121.h5"
store= "../../../DATA/cancer/xai/cancer/"
point=[1,11]
# run train of main segmentation
mn.xai(image,model,store,point,model_name='densenet121',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180,backbone="none",pca='on')

mn1=main.main('mhl_cancer',["normal","cancer"])
image1= "../../../DATA/cancer/2/"              #Copy of BJRas2_c1_23.tif"
model1="../ph_pipeline/Model/weights_binary_cancer_densenet121.h5"
store1= "../../../DATA/cancer/xai/normal/"
point1=[1,11]
# run train of main segmentation
mn1.xai(image1,model1,store1,point1,model_name='densenet121',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180,backbone="none",pca='on')

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
#mn3=main.main('binary_cancer',["normal","cancer"])
#image3= "../../../DATA/cancer/1/Copy of BJprimary4_c1-3_6.tif"
#model3="../ph_pipeline/Model/weights_binary_cancer_MHL.h5"
#store3= "../../../DATA/cancer/xai/healthy_den2.png"
#point3=[1,10]
# run train of main segmentation
#mn3.xai(image3,model3,store3,point3,model_name='MHL',height=512,width=512,channels=3,classes=2,case='mhl_cancer',batch=1,label=[1,0],rot=180)

#mn2=main.main('binary_cancer',["normal","cancer"])
#image12= "../../../DATA/cancer/2/Copy of BJRas1_c1_22.tif"
#model12="../ph_pipeline/Model/weights_binary_cancer_RGB.h5"
#store12= "../../../DATA/cancer/xai/cancer_den2.png"
#point12=[1,13]
# run train of main segmentation
#mn2.xai(image12,model12,store12,point12,model_name='RGB',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180)

# name of weights store of main segmentation
#mn=main.main('ph_200',['cloudy','rain','shine','sunrise'])
#mn4=main.main('binary_cancer',["normal","cancer"])
#image4= "../../../DATA/cancer/1/Copy of BJprimary4_c1-3_6.tif"            #BJprimary_c1-2.tif"
#model4="../ph_pipeline/Model/weights_binary_cancer_vgg16.h5"
#store4= "../../../DATA/cancer/xai/healthy_vgg2.png"
#point4=[1,11]
## run train of main segmentation
#mn4.xai(image4,model4,store4,point4,model_name='vgg16',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[1,0],rot=180)

#mn14=main.main('binary_cancer',["normal","cancer"])
#image14= "../../../DATA/cancer/2/Copy of BJRas1_c1_22.tif"  #Copy of BJRas1_c1_50.tif"
#model14="../ph_pipeline/Model/weights_binary_cancer_vgg16.h5"
#store14= "../../../DATA/cancer/xai/cancer_vgg2.png"
#point14=[1,11]
# run train of main segmentation
#mn14.xai(image14,model14,store14,point14,model_name='vgg16',height=512,width=512,channels=3,classes=2,case='binary_cancer',batch=1,label=[0,1],rot=180)
