#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
import numpy as np
import sys
import os
import cv2
import argparse
import logging
import tensorflow as tf
from ph_pipeline import  create_net, run_model, config, datasetnet, store_model, handle_data
from ph_pipeline.datasetnet import *
#from ph_pipeline.model import roi_net, main_net
import pylab 
import matplotlib.cm as cm
#from ph_pipeline.regularization import data_augmentation
#from ph_pipeline.gan_synthetic import *
import time
from tensorflow.keras import utils
#from ph.CycleGAN.CGAN_model import cyclegan
#from ph.model import crossovernet
import os
#from scipy.ndimage import convolve
#from sklearn import linear_model, datasets, metrics
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
#from sklearn.metrics.classification import accuracy_score



class patch_build(object):

	def __init__ (self, rmn,mmn) :

		args = config.parse_arguments()
		self.param_reg=0.0001
		#self.loss_main=args.loss_main
		#self.loss_roi=args.loss_roi
		#self.pretrain_window=args.pretrain_window
		self.roi_model_name=rmn
		self.main_model_name=mmn
		self.batch_size=args.batch_size
		self.bst=args.batch_size_test
		#self.gan_train_directory=args.gan_train_directory
		self.gancheckpoint='checkpoint'
		self.original_image_shape_roi=args.original_image_shape_roi
		self.original_image_shape_main=args.original_image_shape
		#self.restore_from_jpg_tif=args.restore_image
		self.data_augm=args.data_augm
		self.data_augm_classic=args.data_augm_classic
		self.store_model_path=args.store_txt
		self.validation=args.validation	
		#self.gan_synthetic=args.gan_synthetic
		#self.gan_loops=args.num_synthetic_images
		#self.main_activation=args.main_activation
		self.main_model=args.main_model
		self.STORE_PATH1=args.store_data_test
		self.STORE_PATH2=self.STORE_PATH1+'/ROI/test/'
		self.data_extention=args.data_extention
		self.counter_extention=args.counter_extention
	
	def build_data_structure(self,clas):
		directory=self.STORE_PATH1+'/'+clas
		if not os.path.exists(directory):
			os.mkdir(directory)


	def patch_extract(self,class_list='off', patch_export='on',width=32,height=32, labels=[0,1], threshold=0.75, tr_class=0.0001,store_format='jpg',split_data=1):
		ο=1
		οο=1
		y_array=[]	
	#	dsn=datasetnet.datasetnet("train","main"i,split=100)
                #set the shape of roi analysis for the output lab
		print("Patch analysis pre-processing!")
		for i in create_dataset_itter(split=split_data,analysis='train',path_case='main'):
			X, X_total, Yout, contour_mask, storetxt, itter = i[0],i[1],i[2],i[3],i[4],i[5]
			print(X.shape,Yout.shape,len(labels))
			hd=handle_data.handle_data(X,Yout,'main')
			Y=hd.binary_masks(Yout,len(labels))
			print('check one')
			print(Y.shape)
			totalX, height1, width1 , channels= X.shape
			_, height2, width2, classes = Y.shape
			number=int(height1/height)
			#create the analysis net
			if class_list=='on':
				tr=tr_class
#				for run in range(0,classes):
#				self.build_data_structure('class_'+str(run))
				for i in range(Y.shape[0]):
					y=Y[i,:,:,:]
					if i==0:
						print("values: ",y)
						print("shape: ",np.array(y).shape)
						print("classes: ",classes)
					#clas=0# multi-class
					clas=[] #multi-label
					countc=0
					label_size=[]
					#no multi-label casses only multi-class
					tr_max=int(tr*height2*width2)
					for z in range(1,classes):
						y1=y[:,:,z]
						occur_clas = np.count_nonzero(y1==1)
						label_size.append(occur_clas)
						# all the labels multi-label case
						if label_size[countc]>=tr_max:
						 	clas.append(z)	
	
						# max pool threshold too biased only for multi-class
						#if label_size[countc]>=tr_max:
						#	clas=z	
						#	tr_max=label_size[countc] #verify to have always the max label
						#	#print("achieved class: ",clas)
						
						countc=countc+1
					clas=np.array(clas)
					run=[]
					for io in range(len(clas)): 
						run.append(str(clas[io]))
					class_name=('_'.join(run))
					self.build_data_structure('class_'+class_name)
					x=X[i,:,:,:]
					str3=self.STORE_PATH1 +'/class_'+class_name+ '/Image_%s_%s.%s' % (i,class_name,store_format)
					cv2.imwrite(str3,x)
						#print('write image in class: ',str(clas))
 
			if patch_export=='on':
				split_gpu=1
				start=0
				end=int(totalX/split_gpu)
				Y=np.reshape(Y,[totalX,height2,width2,classes])
				X=np.reshape(X,[totalX,height1,width1,channels])

				for u in range(0,split_gpu):
					print( "The batch number of pre-processing is: ",u)

					shape= self.original_image_shape_main
					thr=0.5 
					xs=int(height*thr) #int(0.8*width)
					ys=int(width*thr)  #int(0.8*height)
					
					Xtotal, Xtot = [], []
					Xtotal=X[start:end]
					y_array=[]
					Ytotal, Ytot = [], []
					Ytotal=Y[start:end]
					start=end
					end=end+int(totalX/split_gpu)
					print(Xtotal.shape,Ytotal.shape)
					patch_size_h = [1, height, width, 1] #channels]
					patch_size_w= [1, height, width,1] #classes]
		
					Xtotal=tf.convert_to_tensor(Xtotal,dtype=tf.int32)
					Ytotal=tf.convert_to_tensor(Ytotal,dtype=tf.int32)
		
					x=tf.image.extract_patches(images=Xtotal, sizes=patch_size_h, strides=[1,xs,ys,1], rates=[1,1,1,1], padding='VALID')
					#print('here')
					#batch_u=0
					#split=10
					#for u in range(1,split):
					#	oldbatch=batch_u
					#	batch_u=batch_u+int((Ytotal.shape[0]/split))
					#	if batch_u>Ytotal.shape[0]:
					#		batch_u=Ytotal.shape[0]
					y1=tf.image.extract_patches(images=Ytotal, sizes=patch_size_w, strides=[1,xs,ys,1], rates=[1,1,1,1], padding='VALID')
					y_array.append(y1)
						
					y=np.array(y_array)
					y=tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])	
					number1=int(x.shape[3]/channels)
					number2=int(y.shape[3]/classes)
					print(number1,number2)
					print('reach this point...')
					print(x.shape,y.shape,channels,classes,number1, number2)
					xphs=tf.reshape(x,[number1*Xtotal.shape[0],x.shape[1],x.shape[2],channels])
					yout=tf.reshape(y,[number2*Ytotal.shape[0],y.shape[1],y.shape[2],classes])
		
				# Check if the threshold number of the patch is == with only one label if yes save if not discard
	
					y_save=[]
					x_save=[]
					label_size=[]
					print(number2*yout.shape[0],yout.shape[0])
					number2=1 #test version
					for i in range(0,number2*yout.shape[0],yout.shape[0]):
						Xphs=(xphs[i:i+yout.shape[0],:,:,:])
						Yphs=yout[i:i+yout.shape[0],:,:,:]
						#print(Yphs.shape)
						for o in range(0,Yphs.shape[0]):
							patch_window=np.array(Yphs[o,:,:,:])
							count=0
							#print('check of label:')
							#print(patch_window.shape)
							for z in labels:				
								occur = np.count_nonzero(patch_window==z)
								label_size.append(occur)
								#print(occur)
								#print(threshold*height*width/4)
								if label_size[count]>=int(threshold*height*width/4):
									Ynp=Yphs[o,:,:,:]
									Ynp=np.where(Ynp!=z,z,z)
									#print('in..')
									#print(Ynp,z)
									y_save.append(Ynp)
									Xnp=Xphs[o,:,:,:]
									x_save.append(Xnp)
								count=count+1
							print(np.array(y_save).shape[0])
							for k in range(0,np.array(y_save).shape[0]):
								y=np.array(y_save[k])
								x=np.array(x_save[k])
								#print(y.shape,x.shape)
								str2=self.STORE_PATH2 + 'X_%s_%s_%s.%s' % (itter,i,k,'jpg')           
								str3=self.STORE_PATH2 + 'Y_%s_%s_%s.%s' % (itter,i,k,'jpg')
								#print(y[12,12,:])
								cv2.imwrite(str3,x)
								#print('image ')
								#print(k)
								#print('saved')
					print('finish split step')						


#	def rebuild_image(self,main_net='on',width=284,height=284,mask='epi'):


