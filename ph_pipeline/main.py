#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
import numpy as np
import sys
import os
import cv2
import csv
import argparse
import logging
import tensorflow as tf
from ph_pipeline import  create_net, run_model, config, datasetnet, store_model, handle_data
from ph_pipeline import main_net, train_vit, vit
import configparser

#from tensorflow.keras import datasets
from ph_pipeline.vit import ViT
from ph_pipeline.train_vit import Trainer, TrainerConfig

from pylab import *
from ph_pipeline import regularization
from ph_pipeline.regularization import data_augmentation
from ph_pipeline import proto_train, proto_setup, proto_setup_eval
from ph_pipeline.proto_setup import train
from ph_pipeline.proto_setup_eval import eval

import time
from tensorflow.keras import utils
from PIL import Image

import pandas as pd
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, roc_auc_score, f1_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from ph_pipeline.XAI.GradCam.xai_resnet50 import *
from ph_pipeline.XAI.GradCam import *
from ph_pipeline.XAI.GradCam.pca_gcam import *
from itertools import cycle
import matplotlib.pyplot as plt





class main(object):

	def __init__ (self, mmn,labels_print=['abnormal','covid','normal2']) :
		args = config.parse_arguments()
		self.width=args.width
		self.height=args.height
		self.channels=args.channels
		self.classes=args.classes
		self.loss_main=args.loss_main
		self.op=args.m_optimizer
		self.main_model_name=mmn
		self.batch_size=args.batch_size
		self.original_image_shape_roi=args.original_image_shape_roi
		self.original_image_shape_main=args.original_image_shape
		self.data_augm=args.data_augm
		self.data_augm_classic=args.data_augm_classic
		self.store_model_path=args.store_txt
		self.validation=args.validation	
		self.cv=args.crossval_cycle
		self.main_activation=args.main_activation
		self.data_extention=args.data_extention
		self.counter_extention=args.counter_extention	
		self.type_analysis=args.type_analysis
		self.main_model=args.main_model
		self.epochs=args.epochs_main
		self.n=args.ngpu
		self.mnm=mmn
		self.Y=0
		self.X=0		 
		self.labels1=labels_print




##################################################################################################################################


	def train_proto(self,case):

		if case=='train':
			descript='Run training'
			conf='TRAIN'
		else:
			descript='Run evaluation'
			conf='EVAL'
		parser = argparse.ArgumentParser(description=descript)
		parser.add_argument("--config", type=str, default="./covid19.config",help="Path to the config file.")
        	# Run training
		args = vars(parser.parse_args())
		config = configparser.ConfigParser()
		config.read(args['config'])
		config = proto_train.preprocess_config(config[conf])
		if case=='test':
			eval(config)        
		else:
			train(config)



	def train_run(self,backbone_n="none",name='none'):

		'''
			Run model create the main segmentation model of image
		'''
		#TODO tine learning-train model tha already stored with weigths$
		#Main-train
		#Crop the label of the myo in ROI shape to learn the main model
		#cretate dataset

		dsn=datasetnet.datasetnet('train')
		X, X_total, Y, contour_mask, storetxt = dsn.create_dataset()
		cn=create_net.create_net(self.main_model)
		Xaug, Yaug= [], []
		classes=np.array(Y).shape[1]
		_, height, width, channels= np.array(X).shape

		if (((self.data_augm == 'True') or (self.data_augm_classic == 'True')) and (self.type_analysis=='SE')):
			for i in data_augmentation(X,Y):
				Xaug.append(i[0])
				Yaug.append(i[1])
			image, mask= np.asarray(Xaug) , np.asarray(Yaug)
			_, _, height, width, channels = image.shape
			print(mask.shape)
			_, _, classes, _ ,_ = mask.shape
		print(height,width,channels,classes,self.labels1)
		#create the analysis net
		model_structure=cn.net([], [], self.main_model_name , height,channels,(classes),width,backbone_n,name) # classes +1 to take into account the background
		print(self.main_model)
		for i in range(len(model_structure)):
			model=model_structure[i]
			rm=run_model.run_model(self.main_model_name,labels_print=self.labels1)
			rm.weight_name=cn.main_model_names_transfer[i]

			if self.main_model=="RGB":
				#rgb_weights = [0.2989, 0.5870, 0.1140]
				#Xgs=np.dot(X[...,:3], rgb_weights)
				#print("grayscale convert rgb image: ",Xgs.shape)
				#Xgs=Xgs/np.max(Xgs)
				#Xgs=np.reshape(Xgs,[Xgs.shape[0],Xgs.shape[1],Xgs.shape[2],1])
				#Xgs=tf.keras.utils.to_categorical(np.asarray(Xgs),(2))
				#Xgs=Xgs[:,:,:,1]
				#Xgs=np.reshape(Xgs,[Xgs.shape[0],Xgs.shape[1],Xgs.shape[2],1])
				model_weights, history_main =rm.run_training(self.loss_main, model, X, Y) 
			else:
				model_weights, history_main =rm.run_training(self.loss_main, model, X, Y)



		
	def train_with_eval(self):

		dsn=datasetnet.datasetnet('train')
		X, X_total, Y, contour_mask, storetxt = dsn.create_dataset()
		cn=create_net.create_net(self.main_model)
		Xaug, Yaug= [], []
		classes=1
		if ((self.data_augm == 'True') or (self.data_augm_classic == 'True')):
			for i in data_augmentation(X,Y):
				Xaug.append(i[0])
				Yaug.append(i[1])
			image, mask= np.asarray(Xaug) , np.asarray(Yaug)
			ya, y, height, width, channels = image.shape
			print(mask.shape)
			if (self.type_analysis=='SE'):
				_, _, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				xa,x, classes= mask.shape
			y1=y*ya
			x1=xa*x
		else:
			image, mask= np.asarray(X) , np.asarray(Y)
			print(image.shape, mask.shape)
			y1, height, width, channels = image.shape
			if (self.type_analysis=='SE'):
				x1, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				x1, classes= mask.shape

		print(height,width,channels,classes)
                #create the analysis net
		 #image=image.reshape(y1,height*width*channels)
		 #mask=mask.reshape(x1,classes)
		print(image.shape,mask.shape)
		
		model_structure=cn.net([], [], self.main_model_name , height,channels,(classes),width) # classes +1 to take into account th$
		model_structure.compile(loss=self.loss_main, optimizer=self.op, metrics=['accuracy'])

		estimators = []
		estimators.append(('standardize', StandardScaler()))
		estimators.append(('mlp', KerasClassifier(build_fn=model_structure, epochs=self.epochs, batch_size=self.batch_size, verbose=0)))
		kfold = StratifiedKFold(n_splits=self.cv, shuffle=True)

		score=["recall", "precision", "roc_auc", "f1", "f1_micro", "f1_macro", "f1_weighted", "f1_samples"]
		results = cross_validate(pipeline, image, mask,scoring=score, cv=kfold, n_jobs=self.n)

		print("Average: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))		
		for i in range(8):
			print("test results %s : %.2f%% " % (score[i], results[score[i]]*100))

		y_pred = cross_val_predict(pipeline, image, mask, cv=kfold,n_jobs=self.n)
		fig, ax = plt.subplots(self.classes, self.classes, figsize=(12, 7))
		vis_arr=np.asarray(multilabel_confusion_matrix(y_pred,mask))
		for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, classes_name):
			print_confusion_matrix(cfs_matrix, axes, label, ["Y", "N"])
			fig.tight_layout()
		plt.savefig(self.store_model_path+'Coeff_Metr_Of_model.png' )
		y_pred_str=map(str,y_pred)
		with open(self.store_model_path+'ypred_out.csv', mode='w') as employee_file:
			employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			employee_writer.writerow(['Predict_class:',y_pred_str])
			

	def test_run(self,path='/jmain02/home/J2AD003/txk56/mxm87-txk56/DATA/'):
		
		dsn=datasetnet.datasetnet('test')
		X, X_total, Y, contour_mask, storetxt = dsn.create_dataset()
		cn=create_net.create_net(self.main_model)
		classes=np.array(Y).shape[1]	
		_, height, width, channels= np.array(X).shape
		self.X=X_total
		self.Y=Y
		labels=self.labels1
		model_structure=cn.net([], [], self.main_model_name , height,channels,(classes),width) # classes +1 to take into account the ba

		if len(model_structure)>1:
			if self.main_model_name=='revgg66':
				model_structure=cn.net([], [], 'REVGG' , height,channels,(classes),width)
			elif self.main_model_name=='denseres171':
				model_structure=cn.net([], [], 'DENRES' , height,channels,(classes),width)
			else:
				print('need more transfer learning networks ;P')
		model=model_structure[0]
		print(X.shape,Y.shape)
		fil=str(self.main_model)
		self.test_result(model, [fil],path,2)

	
	def test_result(self, model_structure, model_name,  path, switch=1):
	
		'''
			Test model to create the main segmentation model of image
		'''	
		classes_name=self.labels1
		sm=store_model.store_model()
		if switch==1:
			dsn_main_test=datasetnet.datasetnet('train')
			mainXtest2,  mainX_total2,mainY2, maincontour_mask2, mainstoretxt2 = dsn_main_test.create_dataset()
			mainXtest=mainX_total2
		if switch==2:
			mainXtest=self.X
			mainY2=self.Y
			print(mainY2.shape)
		
		if self.type_analysis=='CL':
			colors = cycle(['aqua', 'darkorange', 'cornflowerblue','viridis', 'plasma', 'inferno', 'magma', 'cividis'])
			print(len(classes_name))
			print(classes_name)
			print(model_name[0])
			for o in range(len(model_name)): 
				plt.figure()
				fpr, tpr, roc_auc, y_t=sm.load_best_callback(model_structure,mainXtest,mainY2, model_name[o], self.mnm)
				class_number=range(self.classes)
				for i, color in zip(range(self.classes), colors):
					print(roc_auc[i])
					plt.plot(fpr[i], tpr[i],label='ROC curve area = %f of  %s class '  %( roc_auc[i], classes_name[i]))
				plt.plot([0, 1], [0, 1], 'k--')
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('ROC curve characteristics of %s' %model_name[o])
				plt.legend(loc="lower right")
				plt.savefig(path+'ROC_curve_%s.png' % model_name[o])
				
				plt.figure()
				fig, ax = plt.subplots(16, 16) 

				print(y_t.shape,mainY2.shape)
				print(y_t[1,:],mainY2[1,:])
				mainY2=np.reshape(mainY2,(mainY2.shape[0],mainY2.shape[1]))
				y_t=np.asarray(y_t)
				mainY2=np.asarray(mainY2)
				y_t=np.reshape(y_t,(mainY2.shape[0],mainY2.shape[1]))
				print(y_t.shape,mainY2.shape)
				vis_arr=np.asarray(multilabel_confusion_matrix((y_t),mainY2))   
				print(vis_arr.shape)
				print(vis_arr)
				for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, classes_name):
					print(cfs_matrix.shape)
					self.print_confusion_matrix(cfs_matrix, axes, label, ["Y", "N"])
				fig.tight_layout()
				plt.savefig(path+'Coeff_Metr_%s.png' % model_name[o])
				print(y_t.shape,mainY2.shape)


				#for i in range(self.classes):

				y_t1=y_t.astype(int)
				mainY21=mainY2.astype(int)
				print(y_t1.shape,mainY21.shape)
				try:
					ras=roc_auc_score(y_t1,mainY21,average='macro', multi_class='ovr')
				except ValueError:
					ras=0
					pass

				try:
					rasq=roc_auc_score(y_t1,mainY21,average='micro')
				except ValueError:
					rasq=0
					pass
				try:
					rasw=roc_auc_score(y_t1,mainY21,average='weighted')
				except ValueError:
					rasw=0
					pass
				try:	
					rase=roc_auc_score(y_t1,mainY21,average='samples', multi_class='ovr')
				except ValueError:
					rase=0
					pass				
				print("The results of class")
				print ("are:")

				try:

					f1s=f1_score(y_t1,mainY21,average='macro')
				except ValueError:
					f1s=0
					pass
				try:
					f1sq=f1_score(y_t1,mainY21,average='micro')
				except ValueError:
					f1sq=0
					pass
				try:
					f1sw=f1_score(y_t1,mainY21,average='weighted')
				except ValueError:
					f1sw=0
					pass				
				try:
					f1se=f1_score(y_t1,mainY21,average='samples')
				except ValueError:
					f1se=0
					pass
				try:
					pr=precision_score(y_t1,mainY21, average='samples')
				except ValueError:
					pr=0
					pass
				try:
					re=recall_score(y_t1,mainY21, average='samples')
				except ValueError:
					re=0
					pass

				print('RAS macro:')
				print(ras)
				print('RAS micro:')
				print(rasq)				
				print('RAS weighted:')
				print(rasw)
				print('RAS sample:')
				print(rase)	

				print('F1: macro')
				print(f1s)
				print('F1 micro:')
				print(f1sq)
				print('F1 weighted:')
				print(f1sw)
				print('F1 samples:')
				print(f1se)
				print('Prec:')
				print(pr)
				print('Recall:')
				print(re)

				y_pred_str=map(str,y_t)
				ns="/y_pred_out"+"%s.csv" %(model_name[o])
				with open(self.store_model_path+ns, mode='w') as employee_file:
					employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					y_pred_str=zip([map(str,y_t[:,u]) for u in range((np.array(y_t).shape[1]))])
					print("y_pred prediction:")
					for row in y_pred_str:
						for t in row:
							employee_writer.writerow(t)
							print(t)
						
				y_real_str=map(str,mainY2)
				ns2="/y_real_out"+"%s.csv" %(model_name[o])
				with open(self.store_model_path+ns2, mode='w') as employee_file2:
					employee_writer2 = csv.writer(employee_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					y_r_str=zip([map(str,mainY2[:,u]) for u in range((np.array(mainY2).shape[1]))])
					print("y_r_str prediction:")
					for row2 in y_r_str:
						for s in row2:
							employee_writer2.writerow(s)
							print(s)

				
		if self.type_analysis=='SE':

			if mainXtest2==[]:
				print(" no testing ... ")
				print(" Continue ... ")
			else:
				mainpred2, mainmetric2=sm.load_best_callback(mainXtest,mainY2,self.main_model_name,self.main_model_name)
				# store images 
				vo=vision_outputs('main')
				vo.store_test_results(mainmetric2)
				print('Final test loss:', (mainmetric2[0]))
				print('Final test accuracy:', (mainmetric2[1]*100.0))
				print('Final test accuracy:', (mainmetric2[2]*100.0))
				print('Final test accuracy:', (mainmetric2[3]*100.0))
				print(mainpred2.shape)
				# store images 
				length2=len(mainXtest2)
				classes2=self.classes


	def train_vit(self,model_vit='res',shape=64):

		#call cifar10 dataset for fefault test
		#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


		#configuration file extraction
		dsn=datasetnet.datasetnet('train')
		X, X_total, Y, contour_mask, storetxt = dsn.create_dataset()

		print("Start the Cross validation")
		h_d=handle_data.handle_data(X,Y,self.mnm)
		training_augment, train_steps_per_epoch, validation_augment, val_steps_per_epoch =h_d.validation_data()
		Xtr=h_d.X
		Ytr=h_d.Y
		Xte=h_d.Xval
		Yte=h_d.Yval
		len_train=int(Xtr.shape[0])
		len_test=int(Xte.shape[0])
		cn=create_net.create_net(self.main_model)
		Xaug, Yaug= [], []
		classes=1
		if ((self.data_augm == 'True') or (self.data_augm_classic == 'True')):
			for i in data_augmentation(Xtr,Ytr):
				Xaug.append(i[0])
				Yaug.append(i[1])
			image, mask= np.asarray(Xaug) , np.asarray(Yaug)
			_, _, height, width, channels = image.shape
			print(mask.shape)
			if (self.type_analysis=='SE'):
				_, _, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				_,_, classes= mask.shape

		else:

			image, mask= np.asarray(Xtr) , np.asarray(Ytr)
			print(image.shape, mask.shape)
			_, height, width, channels = image.shape
			if (self.type_analysis=='SE'):
 				_, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				_, classes= mask.shape

		train_images=np.array(image)
		train_labels=np.array(mask)
		len_train=(train_images).shape[0]
          
		train_images = tf.cast(train_images.reshape((image.shape[0], channels, shape, shape)),dtype=tf.float32)
		#train_images = train_images / 255.0
		print(train_images.shape)
		train_x = tf.data.Dataset.from_tensor_slices(train_images)
		train_y = tf.data.Dataset.from_tensor_slices(train_labels)

		train_dataset = tf.data.Dataset.zip((train_x,train_y))

		test_images=np.asarray(Xte)
		test_labels=np.asarray(Yte)
		len_test=(test_images).shape[0]

		test_images = tf.cast(test_images.reshape((Xte.shape[0], channels, shape, shape)),dtype=tf.float32)
		#test_images = test_images / 255.0
		print(test_images.shape)
		test_x = tf.data.Dataset.from_tensor_slices(test_images)
		test_y = tf.data.Dataset.from_tensor_slices(test_labels)
		test_dataset = tf.data.Dataset.zip((test_x,test_y))

		tconf = TrainerConfig(max_epochs=100, batch_size=4, learning_rate=0.0000001)
		# sample model config.
		model_config = {"image_size":shape, "patch_size":4, "num_classes":classes, "dim":(2*shape), "depth":3, "heads":4, "mlp_dim":(4*shape), 'channels':channels, 'model_case':model_vit}
		trainer = Trainer(ViT, model_config, train_dataset, len_train, test_dataset, len_test, tconf)
		trainer.train()


		

	def xai(self,image,model,store,point=[7,9],model_name='none',height=256,width=256,channels=1,classes=1,case='test',batch=1,label=[0,0,0,1],rot=0,backbone="none",pca='off',input_n=1):
		if pca=='off':
			viewer(image,model,store,point,model_name,height,width,channels,classes,case,batch,label,rot,backbone)
			print("Finish the xai storing image path store: ",store)
			
		else:
			viewer2(image,model,store,point,model_name,height,width,channels,classes,case,batch,label,rot,backbone,input_n)
			print("Finish the PCA xai storing image path store: ",store)



	

	def load_img_from_fol(self,folder):
		images = []
		for filename in os.listdir(folder):
			img = Image.open(os.path.join(folder,filename)).convert('RGB')
			images.append(img)
		return images


	def print_confusion_matrix(self,confusion_matrix, axes, class_label, class_names, fontsize=14):
		df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
		try:
			heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
		except ValueError:
			raise ValueError("Confusion matrix values must be integers.")
		heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
		heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
		axes.set_xlabel('True label')
		axes.set_ylabel('Predicted label')
		axes.set_title("Confusion Matrix for the class - " + class_label)
		
	
	def attention_map(self,img_path,output_path,model_name,model_path,layer):
		folder=img_path
		images= []
		o=0
		for filename in os.listdir(folder):
			print(filename)
			img=Image.open(os.path.join(folder,filename)).convert('L')
			img=np.array(img)
			img=np.resize(np.array(img),[self.channels, self.height, self.width])
			img=np.reshape(np.array(img),[1, self.channels, self.height, self.width])			
			if o==0:
				images=img
				o=1
			else:
				images=np.append(img,images,axis=0)
		images=np.array(images)
		print(images.shape)
		hd=handle_data.handle_data(images,images,self.main_model_name)
		hd.visualize_class_activation_map(model_name, model_path,output_path,layer)


