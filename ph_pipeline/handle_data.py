#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk


from __future__ import division, print_function
import cv2
from tensorflow.keras import backend as K
from ph_pipeline import config, regularization, handle_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import losses, optimizers, utils, metrics
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from math import ceil
from ph_pipeline import create_net, datasetnet
from ph_pipeline.regularization import  *
import argparse
import logging
import os
import numpy as np
import matplotlib as m
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pylab
from pylab import *
from PIL import Image
from scipy.ndimage import zoom


class handle_data:

	def __init__ (self, X,Y,case) :
		args = config.parse_arguments()
		self.main_model=args.main_model
		self.channels2=1
		self.classes=args.classes
		self.channels=args.channels
		self.case=case
		self.loop=args.max_loops
		self.data_augm='False'
		if args.data_augm=='True' or args.data_augm_classic=='True':
			self.data_augm='True'		
		self.batch_size=args.batch_size
		self.batch_size_test=args.batch_size_test
		self.num_cores=args.num_cores
		self.validation_split=args.validation_split
		self.validation=args.validation	
		self.val_num=args.crossval_cycle
		self.shuffle=args.shuffle
		self.normalize_image=args.normalize
		self.batch_size=args.batch_size
		self.X=X
		self.Y=Y
		self.STORE_PATH=args.store_data_test
		self.data_extention=args.data_extention
		self.type_analysis=args.type_analysis
		bool2=False
		if bool2=='True':
			self.threshold_bin=0.8
		else:
			self.threshold_bin=0.1
		self.width=args.width
		self.height=args.height
		self.Xval=X
		self.Yval=Y

	def validation_data(self):

		print("Start the Cross validation")
		pass_total_validation_augmen, validation_augment= [],[]
		if self.type_analysis=='CL' and self.val_num<=1:
			Xnew, Ynew, Y_train, Y_val, X_val, X_train =  [], [], [], [], [], []
			Xc = []
			split = 0
			split_index=0
			length=0
			for i in range(self.classes):
				print(self.Y.shape)
				O=self.Y[:,(self.classes-i-1)]
				print(O)
				length_of_class=self.Y[length:]
				Xc=self.X[length:]
				length=length+(self.Y[(np.where(O==1))].shape[0]-1)
				length_update=(self.Y[(np.where(O==1))].shape[0]-1)
				print(length)
				print(np.array(length_of_class).shape)
				length_of_class = length_of_class[:length_update]
				length_of_class=np.array(length_of_class)
				print(length_of_class.shape)
				Xc=Xc[:length_update]
				split=int((1-self.validation_split) * (length_of_class).shape[0])
				print(split)
				if i!=0:
					Y_train=np.append(length_of_class[:split],Y_train,axis=0)
					Y_val=np.append(length_of_class[split:],Y_val,axis=0) 
					X_train=np.append(Xc[:split], X_train, axis=0)
					X_val=np.append(Xc[split:], X_val, axis=0)


				else:
					Y_train=(length_of_class[:split])
					Y_val=(length_of_class[split:])
					X_train.append(Xc[:split])
					X_val.append(Xc[split:]) 
					X_val=np.array(X_val)
					X_train=np.array(X_train)
					X_val=np.reshape(X_val,[X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3],X_val.shape[4]])
					X_train=np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3],X_train.shape[4]])	

				print(split, np.array(length_of_class).shape)	
				print(np.array(X_train).shape,np.array(X_val).shape,np.array(Y_train).shape,np.array(Y_val).shape)
				split_index=split_index + (split)
				split=0
				Xc=[]
				length_of_class=[]
				print(split_index)
			X_val=np.array(X_val)
			X_train=np.array(X_train)
			Y_train=np.array(Y_train)
			Y_val=np.array(Y_val)
			print(Y_val.shape,Y_train.shape)
			Xnew=np.append(np.array(X_train),np.array(X_val),axis=0)
			Ynew=np.append(np.array(Y_train),np.array(Y_val),axis=0)
			self.X=[]
			self.Y=[]
			self.X=np.array(Xnew)
			self.Y=np.array(Ynew)
			print(self.X.shape,self.Y.shape)
			val_Xaug, val_Yaug=[],[]
			train_steps_per_epoch = ceil(split_index / self.batch_size)
			val_steps_per_epoch = ceil((len(self.X) - split_index) / self.batch_size)

		if self.type_analysis=='SE' or self.val_num>1:
			split_index = int((1-self.validation_split) * len(self.X))
			train_steps_per_epoch = ceil(split_index / self.batch_size)
			val_steps_per_epoch = ceil((len(self.X) - split_index) / self.batch_size)
			val_Xaug, val_Yaug=[],[]

		print(self.X.shape,self.Y.shape)

		if  self.data_augm == 'True':
			Xtraining_augment_total,Ytraining_augment_total=self.augmented_data(split_index)
			print(self.X.shape,self.Y.shape, np.array(Xtraining_augment_total).shape,np.array(Ytraining_augment_total).shape)
			Xtraining_augment_total=np.array(Xtraining_augment_total)
			if self.type_analysis=='SE':
				Ytraining_augment_total=self.binary_masks(Ytraining_augment_total)
			Ytraining_augment_total=np.array(Ytraining_augment_total)
			if self.type_analysis=='CL':
				print('max value of Y is:')
				print(np.max(Ytraining_augment_total))
				Ytraining_augment_total=np.absolute(np.array(Ytraining_augment_total)/np.max(Ytraining_augment_total))
				print(np.max(Ytraining_augment_total))

			print("Training data structure:")
			print(Ytraining_augment_total.shape, Xtraining_augment_total.shape)
			val_Xaug=(self.X[split_index:])
			val_Yaug=(self.Y[split_index:])
			#suffle the data
			if self.shuffle == 'True':
				valstate = np.random.get_state()
				np.random.set_state(valstate)
				np.random.shuffle(val_Xaug)
				np.random.set_state(valstate)
				np.random.shuffle(val_Yaug)

				valstate2 = np.random.get_state()
				np.random.set_state(valstate2)
				np.random.shuffle(Xtraining_augment_total)
				np.random.set_state(valstate2)
				np.random.shuffle(Ytraining_augment_total)

			training_augment=ImageDataGenerator().flow(Xtraining_augment_total,Ytraining_augment_total, batch_size=self.batch_size)
			#reshape to (total_image,height,weight,channels)
			val_Xaug1, val_Yaug2= np.asarray(val_Xaug) , np.asarray(val_Yaug)
			#val_Xaug1=val_Xaug1.reshape((val_Xaug1.shape[0]*val_Xaug1.shape[1],val_Xaug1.shape[2],val_Xaug1.shape[3],val_Xaug1.shape[4]))

			if self.type_analysis=='CL':
			#	val_Yaug2= val_Yaug2.reshape((val_Yaug2.shape[0]*val_Yaug2.shape[1],val_Yaug2.shape[2]))	
				val_Yaug2=np.absolute(np.array(val_Yaug2)/np.max(val_Yaug2))

			if self.type_analysis=='SE':
				val_Yaug2= val_Yaug2.reshape((val_Yaug2.shape[0]*val_Yaug2.shape[1],val_Yaug2.shape[2],val_Yaug2.shape[3],val_Yaug2.shape[4]))			
				val_Yaug2=self.binary_masks(val_Yaug2)

			validation_augment=ImageDataGenerator().flow(val_Xaug1, val_Yaug2, batch_size=self.batch_size)
			print("Validating data structure:")
			print(val_Yaug2.shape, val_Xaug1.shape)
			# reset the epochs and split		
			train_steps_per_epoch = ceil(len(Xtraining_augment_total)/ self.batch_size)
			val_steps_per_epoch = ceil(len(val_Xaug1) / self.batch_size)
			self.Xval=val_Xaug1
			self.Yval=val_Yaug2
			self.X=Xtraining_augment_total
			self.Y=Ytraining_augment_total
			print(train_steps_per_epoch, val_steps_per_epoch)
			
		else:	
			Y=self.Y
			if self.type_analysis=='SE':
				Y=self.binary_masks(self.Y)
				Y=np.absolute(np.array(Y)/np.max(Y))
			Xtraining_augment_total=np.asarray(self.X[:split_index])
			Ytraining_augment_total=np.asarray(Y[:split_index])
			Xv=np.asarray(self.X[split_index:])
			Yv=np.asarray(Y[split_index:])
			if self.shuffle == 'True':
				valstate = np.random.get_state()
				np.random.set_state(valstate)
				np.random.shuffle(Xv)
				np.random.set_state(valstate)
				np.random.shuffle(Yv)

				valstate2 = np.random.get_state()
				np.random.set_state(valstate2)
				np.random.shuffle(Xtraining_augment_total)
				np.random.set_state(valstate2)
				np.random.shuffle(Ytraining_augment_total)
			training_augment=ImageDataGenerator().flow(Xtraining_augment_total,Ytraining_augment_total, batch_size=self.batch_size)
			validation_augment=ImageDataGenerator().flow(Xv, Yv, batch_size=self.batch_size)
			# reset the epochs and split
			self.Yval=Yv
			self.Xval=Xv
			self.X=Xtraining_augment_total
			self.Y=Ytraining_augment_total
		return training_augment, train_steps_per_epoch,validation_augment, val_steps_per_epoch
								


	def no_validation_data (self) :
		print("No cross validation")
		pass_total_validation_augmen, validation_augment= [],[]
		split_index = len(self.X)
		train_steps_per_epoch = ceil(split_index / self.batch_size)
		if  self.data_augm == 'True':
			Xtraining_augment_total,Ytraining_augment_total=self.augmented_data(split_index)
			#Xtraining_augment_total,Ytraining_augment_total=self.connect_data(self.X,self.Y,Xtraining_augment,Ytraining_augment)
			if self.type_analysis=='SE':
				Ytraining_augment_total=self.binary_masks(Ytraining_augment_total)
			if self.type_analysis=='CL':
				Ytraining_augment_total=np.absolute(np.array(Ytraining_augment_total)/np.max(Ytraining_augment_total))
			if self.shuffle == 'True':
				valstate = np.random.get_state()
				np.random.set_state(valstate)
				np.random.shuffle(Xtraining_augment_total)
				np.random.set_state(valstate)
				np.random.shuffle(Ytraining_augment_total)


			Ytraining_augment_total=np.array(Ytraining_augment_total)	
			Xtraining_augment_total=np.array(Xtraining_augment_total)
			if self.main_model=='idenseres171':
				Xtraining_augment_total=np.array([Xtraining_augment_total],[Xtraining_augment_total])
				Ytraining_augment_total=np.array([Ytraining_augment_total],[Ytraining_augment_total])
			training_augment=ImageDataGenerator().flow(Xtraining_augment_total,Ytraining_augment_total, batch_size=self.batch_size)
			print("Training data structure:")
			print(Ytraining_augment_total.shape, Xtraining_augment_total.shape)

		else:
			if self.type_analysis=='SE':
				self.Y=self.binary_masks(self.Y)
			if self.type_analysis=='CL':
				self.Y=np.absolute(np.array(self.Y)/np.max(self.Y))

			self.X=np.array(self.X)
			self.Y=np.array(self.Y)
			if self.shuffle == 'True': 
				valstate = np.random.get_state()
				np.random.set_state(valstate)
				np.random.shuffle(self.X)
				np.random.set_state(valstate)
				np.random.shuffle(self.Y)
			training_augment=ImageDataGenerator().flow(self.X, self.Y, batch_size=self.batch_size)
			# reset the epochs and split		

		return training_augment, train_steps_per_epoch

	def augmented_data (self,split_index_base ) :
		print("Create data augmented images")
		#training augmented data creation				
		Xaug, Yaug, X, Y = [], [], [], []
		split_index2=int(split_index_base*1)
		X=self.X[:split_index_base]
		Y=self.Y[:split_index_base]
		valstate = np.random.get_state()
		np.random.set_state(valstate)
		np.random.shuffle(X)
		np.random.set_state(valstate)
		np.random.shuffle(Y)
		print(Y.shape, X.shape)
		Xaug.append(X)
		Yaug.append(Y)
		for i in data_augmentation(X[:split_index2],Y[:split_index2]):
				Xaug.append(i[0])
				Yaug.append(i[1])
		Xaug1, Yaug1= np.asarray(Xaug) , np.asarray(Yaug)
		print(Xaug1.shape)
		if self.type_analysis=='CL':
			Xaug1=Xaug1.reshape((Xaug1.shape[0]*Xaug1.shape[1],Xaug1.shape[2],Xaug1.shape[3], Xaug1.shape[4]))
			Yaug1= Yaug1.reshape((Yaug1.shape[0]*Yaug1.shape[1], Yaug1.shape[2]))
		if self.type_analysis=='SE':
			Xaug1=Xaug1.reshape((Xaug1.shape[0]*Xaug1.shape[1],Xaug1.shape[2],Xaug1.shape[3],Xaug1.shape[4]))
			Yaug1= Yaug1.reshape((Yaug1.shape[0]*Yaug1.shape[1],Yaug1.shape[2],Yaug1.shape[3],Yaug1.shape[4]))
		#Xaug1=np.append(Xaug1,X)
		#Yaug1=np.append(Yaug1,Y)
		return Xaug1, Yaug1

	def binary_masks(self,Y, mask_num=2):
		print('binary shape of second shape: ',np.array(Y).shape[1])
		#mask_num=self.classes
		if self.data_extention=="nii.gz":
			if (np.max(Y)>2):
				#print(Y[300,:,:,0])
				Yone=np.array(Y)/np.max(Y)
				#print(Yone[300,:,:,0])
				Y=Yone*(mask_num)
				#print(Y[300,:,:,0])
				#Y=int(Y)
				#print(Y[300,:,:,0])
				#Y=(np.array(Y)/np.max(Y))
			#pos_pred_mask = np.array(np.where(Y > self.threshold_bin))
			#a,b,rows_mask,cos_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
			#Y[a,b,rows_mask,cos_mask]=1
			Yout2 = to_categorical(np.asarray(Y),(mask_num+1))
			#print(Yout2[800,:,:,:])
			print('after to_categorical command: ',Yout2.shape)
			Yout=Yout2.reshape(np.array(Yout2).shape[0],np.array(Yout2).shape[1]*np.array(Yout2).shape[4],np.array(Yout2).shape[2],np.array(Yout2).shape[3])
			Yout=Yout.reshape(np.array(Yout).shape[0],np.array(Yout).shape[3],np.array(Yout).shape[2],np.array(Yout).shape[1])
			print(np.array(Yout).shape)


		else:
			mask_number=np.array(Y).shape[1]
			position_mask_end='false'		
			if mask_number>=8:
				mask_number=np.array(Y).shape[3]
				position_mask_end='true'

			if self.case!='main':
				Yout=int((mask_num+1)*np.array(Y)/np.max(Y))

			if self.case=='main':
				if (mask_number==2):
					if position_mask_end=='true':
						if (np.max(Y[:,:,:,0])>3):
						# be sure all the classes are equal to 1 TODO test this 2 classes
							y=Y[:,:,:,0]
				#			Y=int((mask_num+1)*np.array(y)/np.max(y)) 
							Y=np.array(Y[:,:,:,0])/np.max(Y[:,:,:,0])
						pos_pred_mask = np.array(np.where(Y[:,:,:,0] >= self.threshold_bin))
						a,b,rows_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2]
						Y[a,b,rows_mask,0]=1

						pos_pred_mask = np.array(np.where(Y[:,:,:,0] < self.threshold_bin))
						a1,b1,rows_mask1=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2]
						Y[a1,b1,rows_mask1,0]=0

						# be sure all the classes are equal to 1
						if (np.max(Y[:,:,:,1])>3):
							y=Y[:,:,:,1]
						#	Y=int((mask_num+1)*np.array(y)/np.max(y))
							Y=np.array(Y[:,:,:,1])/np.max(Y[:,:,:,1])

						#second class
						pos_pred_mask = np.array(np.where(Y[:,:,:,1] >= self.threshold_bin))
						a,b,rows_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2]
						Y[a,b,rows_mask,1]=1

						pos_pred_mask = np.array(np.where(Y[:,:,:,1] <self.threshold_bin))
						a2,b2,rows_mask2=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2]
						Y[a2,b2,rows_mask2,1]=0

						Y=np.asarray(Y[:,:,:,0])+np.asarray(Y[:,:,:,1])
						if self.type_analysis=='CL':
							Yout = np_utils.to_categorical(np.asarray(Y),(mask_number))
						if self.type_analysis=='SE':
							Yout = np_utils.to_categorical(np.asarray(Y),(mask_number+1))
						print(np.array(Yout).shape)
						Yout_final=[]
						Y1=[]
						Y2=[]
						Y1=Yout[:,:,:,1].reshape(np.array(Yout).shape[0],np.array(Yout).shape[1],np.array(Yout).shape[2],1)
						Y2=Yout[:,:,:,2].reshape(np.array(Yout).shape[0],np.array(Yout).shape[1],np.array(Yout).shape[2],1)

						Yout_final=np.append(Y1,Y2,axis=3)	
						Yout_final=np.array(Yout_final)
						Yout_final=Yout_final.reshape(np.array(Yout_final).shape[0],np.array(Yout_final).shape[1],np.array(Yout_final).shape[2],2)
						Yout=[]
						Yout=Yout_final
						print(np.array(Yout_final).shape)
						print(np.array(Yout).shape)
					else:
				
						if (np.max(Y[:,0,:,:])>3):
							# be sure all the classes are equal to 1 TODO test this 2 classes 
							Y=np.array(Y[:,0,:,:])/np.max(Y[:,0,:,:])

						#first class
						pos_pred_mask = np.array(np.where(Y[:,0,:,:] >= self.threshold_bin))
						a,b,rows_mask,cos_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a,b,rows_mask,cos_mask]=1

						pos_pred_mask = np.array(np.where(Y[:,0,:,:] < self.threshold_bin))
						a1,b1,rows_mask1,cos_mask1=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a1,b1,rows_mask1,cos_mask1]=0

						# be sure all the classes are equal to 1
						if (np.max(Y[:,1,:,:])>3):
							Y=np.array(Y[:,1,:,:])/np.max(Y[:,1,:,:])
						#second class
						pos_pred_mask = np.array(np.where(Y[:,1,:,:] >= self.threshold_bin))
						a,b,rows_mask,cos_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a,b,rows_mask,cos_mask]=1

						pos_pred_mask = np.array(np.where(Y[:,1,:,:] <self.threshold_bin))
						a2,b2,rows_mask2,cos_mask2=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a2,b2,rows_mask2,cos_mask2]=0

						Y=np.asarray(Y[:,0,:,:])+np.asarray(Y[:,1,:,:])
						if self.type_analysis=='CL':
							Yout = np_utils.to_categorical(np.asarray(Y),(mask_number))
						if self.type_analysis=='SE':
							Yout = np_utils.to_categorical(np.asarray(Y),(mask_number+1))
						Yout=Yout.reshape(np.array(Yout).shape[0],np.array(Yout).shape[2],np.array(Yout).shape[3],np.array(Yout).shape[4])
					
				if (mask_number<=1):
					if (np.max(Y)>2):
						Y=np.array(Y)/np.max(Y)
					if position_mask_end=='false':
						# be sure all the classes are equal to 1 
						pos_pred_mask = np.array(np.where(Y > self.threshold_bin))
						a,b,rows_mask,cos_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a,b,rows_mask,cos_mask]=1
						if self.type_analysis=='CL':
							Yout2 = np_utils.to_categorical(np.asarray(Y))
							print(Yout2.shape)
							Yout2=Yout2.reshape(np.array(Yout2).shape[0],np.array(Yout2).shape[2],np.array(Yout2).shape[3],np.array(Yout2).shape[1]*np.array(Yout2).shape[4])
							Yout=Yout2[:,:,:,0]
							Yout=Yout.reshape(np.array(Yout).shape[0],np.array(Yout).shape[1],np.array(Yout2).shape[2],1)
						if self.type_analysis=='SE':
							Yout2 = np_utils.to_categorical(np.asarray(Y),(mask_number+1))
							Yout=Yout2.reshape(np.array(Yout2).shape[0],np.array(Yout2).shape[2],np.array(Yout2).shape[3],np.array(Yout2).shape[1]*np.array(Yout2).shape[4])
						Yout=Yout.reshape(np.array(Yout).shape[0],np.array(Yout).shape[1],np.array(Yout).shape[2],np.array(Yout).shape[3])
					if position_mask_end=='true':
						# be sure all the classes are equal to 1 
						pos_pred_mask = np.array(np.where(Y > self.threshold_bin))
						a,b,rows_mask,cos_mask=pos_pred_mask[0],pos_pred_mask[1],pos_pred_mask[2],pos_pred_mask[3]
						Y[a,b,rows_mask,cos_mask]=1
						Yout2 = np_utils.to_categorical(np.asarray(Y),(mask_number+1))
						Yout=Yout2.reshape(np.array(Yout2).shape[0],np.array(Yout2).shape[2],np.array(Yout2).shape[3],np.array(Yout2).shape[1]*np.array(Yout2).shape[4])
						Yout=Yout.reshape(np.array(Yout).shape[0],np.array(Yout).shape[1],np.array(Yout).shape[2],np.array(Yout).shape[3])
			# 	print(Yout[1,1,:,:],Yout[3,0,:,:])

		return Yout	


	def connect_data(self, X1,Y1,X2,Y2) :
		print("Connect data_set")
		Xtotal, Ytotal=[],[]
		# add the initial data

		Xtotal=np.append(np.array(X1),np.array(X2),axis=0)
		Ytotal=np.append(np.array(Y1),np.array(Y2),axis=0)
		#reshape to (total_image,height,weight,channels)
		Xtotal1, Ytotal2= np.asarray(Xtotal) , np.asarray(Ytotal)

		#Xtotal1=Xtotal1.reshape((Xtotal1.shape[0]*Xtotal1.shape[1],Xtotal1.shape[2],Xtotal1.shape[3],Xtotal1.shape[4]))
		#Ytotal2= Ytotal2.reshape((Ytotal2.shape[0]*Ytotal2.shape[1],Ytotal2.shape[2],Ytotal2.shape[3],Ytotal2.shape[4]))
		#create an Image data generator
		#suffle the data
		state = np.random.get_state()
		np.random.shuffle(Xtotal1)
		np.random.set_state(state)
		np.random.shuffle(Ytotal2)
		print(Xtotal1.shape,Ytotal2.shape)		
		return Xtotal1,Ytotal2



	def resize_data(self,X,Y):
		r_masks,resize_masks= [], []
		X,Y=np.asarray(X),np.asarray(Y)
		o=0
		while o < 1:#self.classes:
			r_mask_store=resize_masks
			resize_masks, images_resize=[],[]	
			for i in range(len(Y)):
				image_resize = cv2.resize(X[i,:,:,0].reshape(X.shape[1],X.shape[2]), (self.X.shape[1],self.X.shape[2]), cv2.INTER_NEAREST)
				resize_mask = cv2.resize(Y[i,:,:,o].reshape(Y.shape[1],Y.shape[2]), (self.Y.shape[2],self.Y.shape[3]), cv2.INTER_NEAREST)
				resize_mask=np.array(resize_mask).reshape(self.channels2,self.Y.shape[2],self.Y.shape[3])
				resize_masks.append(resize_mask)				
				images_resize.append(image_resize)

			if o>0:	
				r_masks=np.append(r_mask_store,resize_masks,axis=1)
			else:
				r_masks=resize_masks
			o=o+1

		mask_resize=np.array(r_masks).reshape((np.array(r_masks).shape[0], self.channels2,self.Y.shape[2],self.Y.shape[3]))#Y.shape[1]))
		images_resize=np.array(images_resize).reshape((np.array(images_resize).shape[0], self.X.shape[1],self.X.shape[2], self.channels))#Y.shape[1]))

		return images_resize, mask_resize


	def clipped_zoom(self):
		X=self.X
		Y=self.Y	
		r_masks=[]
		resize_masks=[]

		X=np.asarray(X)
		Y=np.asarray(Y)
		o=0
		h = X.shape[2]
		w = h
		zoom_offset=(h-Y.shape[2])//2

		while o < 2:
			r_mask_store=resize_masks
			images_resize=[]
			resize_masks=[]
			for i in range(len(Y)):
				resize_mask = cv2.resize(Y[i,:,:,o].reshape(Y.shape[1],Y.shape[2]), (self.Y.shape[2],self.Y.shape[2]), cv2.INTER_NEAREST)
				image_resize = cv2.resize(X[i,:,:,0].reshape(X.shape[1],X.shape[2]), (self.X.shape[1],self.X.shape[2]), cv2.INTER_NEAREST)

				# Zero-padding
				top=(zoom_offset)
				left=(zoom_offset)
				top2=(h-top)
				left2=(w-left)

				out = np.zeros_like(image_resize)
				out=np.array(out)
				out.astype(double)
				#print(top,top2,left,left2)
				out[top:top2, left:left2] = resize_mask[0:(Y.shape[2]),0:(Y.shape[2])] #zoom(resize_mask,1)

				out=np.array(out).reshape(self.channels2,np.array(out).shape[0],np.array(out).shape[1])
				resize_masks.append(out)
			if o>0:
				r_masks=np.append(r_mask_store,resize_masks,axis=1)
			else:
				r_masks=resize_masks
			o=o+1

		mask_resize=np.array(r_masks).reshape((np.array(r_masks).shape[0], self.channels2, X.shape[1], X.shape[2]))	
		print(np.array(mask_resize).shape)
		print('clipped zoom finished...')
		return np.array(mask_resize)

	def visualize_class_activation_map(self, model_name, model_path, output_path, layer):
		cn=[]
		model=[]
		for i in range(len(model_path)):
			cn.append(create_net.create_net(model_name[i]))
		for o in range(len(model_path)):
			model.append(cn[o].net([], [], self.case, self.height, 1,2,self.width))
		for k in range(len(layer)):
			if len(model_path)==1:
				model_path[k]=model_path[0]
			#cn=create_net.create_net(model_name[k])
			#model=cn[k].net([], [], self.case, self.height, 1,2,self.width)
			model[k].load_weights(model_path[k])
			original_img=np.array(self.X)
			print(original_img.shape)
			original_img=np.reshape(np.array(original_img),[self.X.shape[0], self.channels, self.height, self.width])
			n_i, width, height, channels = original_img.shape
			#Reshape to the network input shape (b, channel, w, h).
			img = np.array([np.transpose(np.float32(original_img), (0,3, 2, 1))])
			img=np.reshape((img),[self.X.shape[0], self.height, self.width,self.channels])
			print(img.shape)       
			#Get the 512 input weights to the softmax.
			class_weights = model[k].layers[-1].get_weights()[0]
			final_conv_layer = model[k].get_layer(layer[k])
			print('test1')
			output_layer=final_conv_layer.get_output_at(-1)

			out1 = K.function([model[k].layers[0].input],[output_layer, model[k].layers[-1].output])
			[out, pred] = out1([img])
			print(out.shape)
			print(img.shape)
			conv_output=np.reshape(out,[out.shape[0], out.shape[1], out.shape[2], out.shape[3]])
			conv_outputs=conv_output
			print(conv_outputs.shape[0:4])
			print(class_weights.shape)
			#Create the class activation map.
			w = 1
			for o in range(n_i):
				cam = np.zeros(dtype = np.float32, shape = [ conv_outputs.shape[1], conv_outputs.shape[2]])
				print(cam.shape)
				for i in range(conv_outputs.shape[3]):
					cam =cam+w*conv_outputs[o, :, :,i]
				cam=cv2.resize((cam), (512, 512),interpolation=cv2.INTER_NEAREST)
				cam=cv2.normalize(cam, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				str3= output_path[k]+'/heatmap_%s'  %(o)
				viridis=plt.get_cmap('viridis',256)
				newcolors = viridis(np.linspace(0, 1, 256))
				red = np.array([248/256, 2/256, 27/256, 1])
				newcolors[235:,:]=red
				newcmp=m.colors.ListedColormap(newcolors)
				cms=[viridis,newcmp]
				fig, axs = plt.subplots(1, 2)
				for [ax, cmap] in zip(axs, cms):
					psm = ax.pcolormesh(cam, cmap=cmap, vmin=0,vmax=1)
					fig.colorbar(psm, ax=ax)
				plt.savefig(str3)
				plt.close('all')



