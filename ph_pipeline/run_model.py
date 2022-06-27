#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk
#Acknowledgement: https://github.com/ece7048/cardiac-segmentation-1/blob/master/rvseg/loss.py


from __future__ import division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from ph_pipeline import config, regularization, handle_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ph_pipeline import create_net, regularization
from tensorflow.keras import losses, optimizers, utils, metrics
#from keras.utils import  np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.models import model_from_json

from math import ceil
import numpy as np
import cv2
import os
import argparse
#from tensorflow.keras.objectives import *
from tensorflow.keras import utils
from ph_pipeline import main, losses_distance
import matplotlib.pyplot as plt



class run_model:
##################################################### INITIALIZATION ##########################################################
	def __init__ (self,mmm='main',labels_print=['abnormal','covid','normal2']) :
		args = config.parse_arguments()
		self.ram= args.ram
		self.cross_validation_number=args.crossval_cycle
		self.ngpu=args.ngpu		
		self.metrics=args.metrics
		self.metrics1=args.metrics1
		self.metrics2=args.metrics2		
		self.batch_size=args.batch_size
		self.batch_size_test=args.batch_size_test
		self.epochs_main= args.epochs_main
		self.num_cores=args.num_cores
		self.path_case=mmm
		self.validation_split=args.validation_split
		self.validation=args.validation	
		self.shuffle=args.shuffle
		self.weights=args.loss_weights
		self.normalize_image=args.normalize
		self.roi_shape=args.roi_shape_roi	
		self.store_model=args.store_txt
		self.weight_name='weights_main'
		self.early_stopping=args.early_stop
		self.monitor=args.monitor_callbacks
		self.mode=args.mode_convert
		self.exponential_decay='False'
		self.step_decay='False'
		self.lrate=args.learning_rate
		self.store_model_path=args.store_txt
		self.width=args.width
		self.height=args.height
		self.channels=args.channels
		self.classes=args.classes
		self.main_model=args.main_model
		self.path=args.datapath+"/"

		if(args.decay==666):
			self.exponential_decay='True'
			args.decay=0
		if(args.decay==999):
			self.step_decay='True'
			args.decay=0
		optimizer_args = {
			'learning_rate':       args.learning_rate,
			'momentum': args.momentum,
			'decay':   args.decay,
			'seed':    args.seed
			}

		#if self.path_case=='main':
		for k in list(optimizer_args):
			if optimizer_args[k] is None:
				del optimizer_args[k]
		optimizer = self.pass_optimizer(args.m_optimizer, optimizer_args)
		self.optimizer=optimizer
		self.epochs=self.epochs_main
		self.labels1=labels_print
		print(self.labels1)
################################################# data_augmentationTRAINING ##############################################################


	def run_training(self,loss_type, model_structure, X, Y):
		# MPU: data paraller mode deep learning 
		# GPU: model paraller mode deep learning 
		#one-hot encoder mask binary labels
		#if self.path_case=='main':
		#	Y = utils.to_categorical(Y,(np.array(Y).shape[1]+1))
		#	Y=Y.reshape(np.array(Y).shape[0],np.array(Y).shape[2],np.array(Y).shape[3],np.array(Y).shape[4])
		#	print(np.array(Y).shape)
		#if self.ram=='GPU':
			#model_structure = multi_gpu_model(model_structure, gpus=self.ngpu)
		#	self.batch_size=self.batch_size*self.ngpu
		#define metrics for training
		metrics_algorithm=self.metrics_generator()
		loss=self.load_loss(loss_type)
		#if loss_type=="RGB_CCE":
		


		model_structure.compile(optimizer=self.optimizer,loss=loss, metrics=metrics_algorithm)
		if self.normalize_image == 'True':
			X=regularization.normalize(X)
		#define callbacks
		self.model_json = model_structure.to_json()
		self.callbacks=self.callbacks_define(self.monitor, self.weight_name)
		#validation data
		cn=create_net.create_net(self.main_model)
		if self.validation=='on':
			for cross_val_num in range(int(self.cross_validation_number)):
				print("cross validation run: ", cross_val_num, "/", int(self.cross_validation_number))
				if cross_val_num!=0:
					model2=cn.net([], [], self.path_case , self.height, self.channels,(self.classes), self.width) 
					model_structure1=model2[0]
					rng_state = np.random.get_state()
					np.random.shuffle(X)
					np.random.set_state(rng_state)
					np.random.shuffle(Y)
					model_structure1.compile(optimizer=self.optimizer,loss=loss, metrics=metrics_algorithm)
					self.callbacks=self.callbacks_define(self.monitor,( self.weight_name+'_'+str(cross_val_num)))
					
				h_d=handle_data.handle_data(X,Y,self.path_case)

				training_augment, train_steps_per_epoch, validation_augment, val_steps_per_epoch =h_d.validation_data()
				if self.ram=='CPU':				
					history = model_structure.fit(training_augment, epochs=self.epochs, steps_per_epoch=train_steps_per_epoch,verbose=1, callbacks=self.callbacks, validation_data=validation_augment, validation_steps=val_steps_per_epoch  )  

				if self.ram=='MPU':  
					self.batch_size=self.batch_size
					history = model_structure.fit(training_augment, epochs=self.epochs, steps_per_epoch=train_steps_per_epoch,verbose=1, callbacks=self.callbacks, validation_data=validation_augment, validation_steps=val_steps_per_epoch,workers=self.num_cores,use_multiprocessing=True)      
				#call test for evaluate results..P.S. need modification this part: TODDO!!!!
				mn=main.main(self.path_case,labels_print=self.labels1)
				print(h_d.Xval.shape,h_d.Yval.shape)
				mn.X=np.array(h_d.Xval)
				mn.Y=np.array(h_d.Yval)
				print('test output')
				#print(mn.X[1,:,:,1], mn.Y[1,:])
				print(self.labels1)
				filepath=str(self.weight_name+'_'+str(cross_val_num))
				file=str(self.weight_name)
				if cross_val_num!=0:
					mn.test_result(model_structure, [filepath],self.path,2)
				else:
					mn.test_result(model_structure, [file],self.path,2)
		else:
			h_d2=handle_data.handle_data(X,Y,self.path_case)
			training_augment, train_steps_per_epoch =h_d2.no_validation_data()
			if self.ram=='CPU':
				history = model_structure.fit(training_augment, epochs=self.epochs, steps_per_epoch=train_steps_per_epoch, verbose=1, callbacks=self.callbacks)   
			if self.ram=='MPU':
				self.batch_size=self.batch_size 
				history = model_structure.fit(training_augment, epochs=self.epochs, steps_per_epoch=train_steps_per_epoch, verbose=1, callbacks=self.callbacks,workers=self.num_cores,use_multiprocessing=True)   
			mn=main.main(self.path_case, labels_print=self.labels1)
			mn.Y=np.array(h_d2.Y)
			mn.X=np.array(h_d2.X)
			file=str(self.weight_name)
			print(self.labels1)
			print(h_d2.X.shape,h_d2.Y.shape)
			mn.test_result(model_structure, [file],self.path,1)

		self.print_history(history, path=self.path)
		return model_structure, history

##################################################### TESTING ##################################################################

	def print_history(self,history,path):
		# summarize history for accuracy
		plt.figure()
		auc=history.history.keys()
		print(auc)
		
		plt.plot(history.history['auc'])
		plt.plot(history.history['val_auc'])

		plt.title('model accuracy metric')
		plt.ylabel('AUC')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(path+'Train_Validation_AUC_curve_%s.png' % self.main_model)

		plt.figure()
		plt.plot(history.history['recall'])
        
		plt.plot(history.history['val_recall'])
		plt.title('model recall metric')
		plt.ylabel('Recall')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(path+'Train_Validation_Recall_curve_%s.png' % self.main_model)

	
		plt.figure()
		plt.plot(history.history['precision'])
		plt.plot(history.history['val_precision'])
		plt.title('model precision metric')
		plt.ylabel('Precision')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(path+'Train_Validation_Precision_curve_%s.png' % self.main_model)

		# summarize history for loss
		plt.figure()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(path+'Train_Validation_loss_curve_%s.png' % self.main_model)

                # summarize history for loss
		plt.figure()
		plt.plot(history.history['f1_m'])
		plt.plot(history.history['val_f1_m'])
		plt.title('model f1')
		plt.ylabel('F1 metric')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig(path+'Train_Validation_dice_curve_%s.png' % self.main_model)




	def run_testing(self,loss_type,model_structure, X, Y):
		# set GPU or CPU
		if self.ram=='GPU':
			model_structure = multi_gpu_model(model_structure, gpus=self.ngpu)
			self.batch_size_test=self.batch_size_test*self.ngpu

		if self.normalize_image == 'True':
			X=regularization.normalize(X)

		if self.shuffle == 'True':
			# shuffle images and masks in parallel
			rng_state = np.random.get_state()
			np.random.shuffle(X)
			np.random.set_state(rng_state)
			np.random.shuffle(Y)
		if self.fourier=="on":
			hd=handle_data.handle_data(X,Y,self.path_case)
			X,Y=hd.fourier_convert_data (X,Y )
			Y=hd.binary_masks(Y)
			#Y=Y.reshape(Y.shape[0],Y.shape[3],Y.shape[2],Y.shape[1])
			print(np.array(Y).shape)

		if (self.path_case=='main' ):
			print(np.array(Y).shape)
			hd=handle_data.handle_data(X,Y,self.path_case)
			Y=hd.binary_masks(Y)
			print(np.array(Y).shape)
		Xtest=X
		Ytest=Y
		#set metric and loss to model 
		#define metrics for testing
		metrics_algorithm=self.metrics_generator()

		loss=self.load_loss(loss_type)
		model_structure.compile(optimizer=self.optimizer,loss=loss, metrics=metrics_algorithm)
		#predict the outputs
		batch_size=self.batch_size_test
		test_generator=ImageDataGenerator().flow(x=Xtest, batch_size=batch_size, shuffle=False )
		y_pred = model_structure.predict_generator(test_generator,steps=np.ceil(len(X)/batch_size))

		hd1=handle_data.handle_data(Xtest,y_pred,self.path_case)

		y_pred=hd1.binary_masks(y_pred)

		hd_final=handle_data.handle_data(Xtest,y_pred,self.path_case)
		if (self.path_case=='main' and self.fourier=="off"):
			y_pred=hd_final.clipped_zoom()
			
		#evaluate the results
		metric=model_structure.evaluate(x=Xtest, y=Ytest, batch_size=batch_size, verbose=1, sample_weight=None, steps=None)

		return y_pred, metric 
############################################ metric, loss, optimizzer choices, callbacks################## 

	def callbacks_define(self,monitor,weight_name):
		# call backs of the weights
		filepath=str(self.store_model + "/weights_%s_%s.h5" % (self.path_case,weight_name)) #hdf5
		print(filepath)
		file_name=str(self.store_model + "/weights_%s_%s" % (self.path_case,weight_name))
		monitor=monitor
		#self.model_json = model.to_json()
		
		with open(file_name + ".json", "w") as json_file:
			json_file.write(self.model_json)

		checkpoint = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=str(self.mode))
		callbacks = [checkpoint]

		if self.early_stopping =='True':
			stop_points=EarlyStopping(monitor=monitor, min_delta=0.0000001, patience=10, verbose=1, mode=str(self.mode), baseline=None, restore_best_weights=True)
			callbacks = [checkpoint, stop_points]
		if self.exponential_decay=='True':
			 # loss_history =LossHistory()
			lrate= LearningRateScheduler(schedule=self.exp_decay, verbose=1)
			callbacks=[checkpoint,lrate]
		if self.step_decay=='True':
			lrate= LearningRateScheduler(schedule=self.step_epoch, verbose=1)
			callbacks=[checkpoint,lrate]
		return callbacks


	def exp_decay(self,epoch):
		k=7.5
		lrate=self.lrate*np.exp(-k*(epoch/self.epochs_main))
		return lrate

	def step_epoch(self,epoch):
		lrate= self.lrate if epoch<(self.epochs_main*0.5) else self.lrate*(self.epochs_main-epoch)
		return lrate

	def pass_optimizer(self,optimizer_name, optimizer_args):
		optimizers = {
		'sgd': SGD,
		'rmsprop': RMSprop,
		'adagrad': Adagrad,
		'adadelta': Adadelta,
		'adam': Adam,
		'adamax': Adamax,
		'nadam': Nadam,
		}
		if optimizer_name not in optimizers:
			raise Exception("Unknown optimizer ({}).".format(optimizer_name))
		return optimizers[optimizer_name](**optimizer_args)

	def pass_metric(self,metric_name):
		if (metric_name == "customized_loss"):
			return self.customized_loss 
		if (metric_name == "weighted_categorical_crossentropy"):				
			return self.weighted_categorical_crossentropy
		if (metric_name == "jaccard_loss"):
			return self.jaccard_loss
		if (metric_name == "hard_jaccard"):
			return self.hard_jaccard
		if (metric_name == "soft_jaccard"):
			return self.soft_jaccard
		if (metric_name == "sorensen_dice_loss"):
			return self.sorensen_dice_loss
		if (metric_name == "hard_sorensen_dices"):
			return self.hard_sorensen_dice
		if (metric_name == "soft_sorensen_dice"):
			return self.soft_sorensen_dice
		if (metric_name == "mean_squared_error"):
			return mean_squared_error
		if (metric_name == "kullback_leibler_divergence"):
			return kullback_leibler_divergence
		if (metric_name == "binary_crossentropy"):
			return binary_crossentropy
		if (metric_name == "sparse_categorical_crossentropy"):
			return sparse_categorical_crossentropy
		if (metric_name == "categorical_crossentropy"):
			return categorical_crossentropy
		if (metric_name=="categorical_accuracy"):
			return metrics.categorical_accuracy
		if (metric_name=="accuracy"):
			return  tf.keras.metrics.Accuracy()
		if (metric_name == "precision"):
			return keras.metrics.Precision()
		if (metric_name == "f1"):
			#f1=2*((self.pass_metric('precision')*self.pass_metric('recall')) / (self.pass_metric('precision')+self.pass_metric('recall')+K.epsilon()))
			return self.f1_m
		if (metric_name == "recall"):
			return keras.metrics.Recall()
		if (metric_name == "sparce_categorical_accuracy"):
			return metrics.sparce_categorical_accuracy
		if (metric_name == "dice"):
			return self.dice
		if (metric_name == "log_dice"):
			return self.log_dice
		if (metric_name == "log_jaccard"):
			return self.jaccard
		if (metric_name == "Hausdorff"):
			return self.loss_Hausdorff_distance
		if (metric_name == "binary_accuracy"):
			return metrics.binary_accuracy
		if (metric_name == "AUCROC"):
			return keras.metrics.AUC(curve='ROC')
		if (metric_name == "AUCPR"):
			return keras.metrics.AUC(curve='PR')
		if metric_name==" " :
			raise Exception("None metric ({}).".format(metric_name))


	def load_loss(self, loss_check):
		if (loss_check == ""):
			def lossfunction(y_true,y_pred):
				return binary_crossentropy(y_true, y_pred)
		if (loss_check == "customized_loss"):
			def lossfunction(y_true, y_pred):
				return self.customized_loss(y_true, y_pred) 
		if (loss_check == "weighted_categorical_crossentropy"):
			def lossfunction(y_true, y_pred):				
				return self.weighted_categorical_crossentropy(y_true, y_pred)
		if (loss_check == "jaccard_loss"):
			def lossfunction(y_true, y_pred):	
				return self.jaccard_loss(y_true, y_pred)
		if (loss_check == "hard_jaccard"):
			def lossfunction(y_true, y_pred):
				return self.hard_jaccard(y_true, y_pred)
		if (loss_check == "soft_jaccard"):
			def lossfunction(y_true, y_pred):
				return self.soft_jaccard(y_true, y_pred)
		if (loss_check == "sorensen_dice_loss"):
			def lossfunction(y_true, y_pred):
				return self.sorensen_dice_loss(y_true, y_pred)
		if (loss_check == "hard_sorensen_dices"):
			def lossfunction(y_true, y_pred):
				return self.hard_sorensen_dice(y_true, y_pred)
		if (loss_check == "soft_sorensen_dice"):
			def lossfunction(y_true, y_pred):
				return self.soft_sorensen_dice(y_true, y_pred)
		if (loss_check == "mean_squared_error"):
			def lossfunction(y_true, y_pred):
				return mean_squared_error(y_true, y_pred)
		if (loss_check == "kullback_leibler_divergence"):
			def lossfunction(y_true, y_pred):
				return kullback_leibler_divergence(y_true, y_pred)
		if (loss_check == "binary_crossentropy"):
			def lossfunction(y_true, y_pred):
				return binary_crossentropy(y_true, y_pred)
		if (loss_check == "sparse_categorical_crossentropy"):
			def lossfunction(y_true, y_pred):
				return sparse_categorical_crossentropy(y_true, y_pred)
		if (loss_check == "categorical_crossentropy"):
			def lossfunction(y_true, y_pred):
				return categorical_crossentropy(y_true, y_pred)
		if (loss_check == "Hausdorff_loss"):
			def lossfunction(y_true, y_pred):
				return self.loss_Hausdorff_distance(y_true, y_pred)
		if(loss_check=="dice_cross_entropy"):
			def lossfunction(y_true,y_pred):
				return self.dice_cross_entropy(y_true,y_pred)
		if(loss_check=="binary_dice_cross_entropy"):
			def lossfunction(y_true,y_pred):
				return self.binary_dice_cross_entropy(y_true,y_pred)
		if(loss_check=="dice_loss"):	
			def lossfunction(y_true,y_pred):
				return self.dice_loss(y_true,y_pred)
		if(loss_check=="log_dice_loss"):	
			def lossfunction(y_true,y_pred):
				return self.log_dice_loss(y_true,y_pred)

		if(loss_check=="jaccard_loss"):	
			def lossfunction(y_true,y_pred):
				return self.jaccard_loss(y_true,y_pred)

		if(loss_check=="sscill"):	
			def lossfunction(y_true,y_pred):
				return self.softmax_sparse_crossentropy_ignoring_last_label(y_true,y_pred)

		if(loss_check=="SS_DL"):
			def lossfunction(y_true,y_pred):
				return self.SS_DL(y_true,y_pred)

		if(loss_check=="SS_DCE"):
			def lossfunction(y_true,y_pred):	
				return self.SS_DCE(y_true,y_pred)

		if(loss_check=="SS_CE"):
			def lossfunction(y_true,y_pred):
				return self.SS_CE(y_true,y_pred)

		if(loss_check=="SS_ba"):
			def lossfunction(y_true,y_pred):
				return self.SS_binary_accuracy(y_true,y_pred)

		if (loss_check=="cgan"):
			def lossfunction(y_true,y_pred):
				return self.loss_gan(y_true,y_pred)
		if (loss_check=="scgan"):
			def lossfunction(y_true,y_pred):
				return self.simple_loss_gan(y_true,y_pred)
		return lossfunction
#
	def metrics_generator(self):
		self.metric4='f1'
		metrics_algorithm = [self.pass_metric(self.metrics), self.pass_metric(self.metrics1), self.pass_metric(self.metrics2),self.pass_metric(self.metric4)]
		return metrics_algorithm

###################################### METRICS MEASUREMENTS ####################################################################
	def dice(self,y_true, y_pred):
		batch_dice_coefs = self.hard_sorensen_dice(y_true, y_pred, axis=[1, 2])
		dice_coefs = K.mean(batch_dice_coefs, axis=0)
		#w = K.constant(self.weights)/sum(self.weights)
		return dice_coefs[0]  

	def log_dice(self,y_true, y_pred, smooth=100):
		log_batch_dice_coefs = self.log_dice_loss_core(y_true, y_pred, axis=[1, 2])
		log_dice_coefs = K.mean(log_batch_dice_coefs, axis=0)
		#w = K.constant(self.weights)/sum(self.weights)
		return log_dice_coefs[0] 

	def jaccard(self,y_true, y_pred):
		batch_jaccard_coefs = self.hard_jaccard(y_true, y_pred, axis=[1, 2])
		jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
		#w = K.constant(self.weights)/sum(self.weights)
		return jaccard_coefs[0] 

	def recall_m(self, y_true, y_pred):
		#print(y_pred.shape)
		#print(y_pred.shape)
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0.0, 1.0)))
		recall = true_positives/(possible_positives + K.epsilon())
		return recall

	def precision_m(self, y_true, y_pred):
		#print(y_pred.shape)
		#print(y_pred.shape)
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0.0, 1.0)))
		precision = true_positives/(predicted_positives + K.epsilon())
		return precision

	def f1_m(self, y_true, y_pred):
		print(y_pred.shape)
		precision = self.precision_m(y_true, y_pred)
		recall = self.recall_m(y_true, y_pred)
		return 2*((precision*recall)/(precision+recall+K.epsilon()))			
###################################### Keras losses #################################################################### 


	def soft_sorensen_dice(self,y_true, y_pred, axis=None, smooth=100):
		intersection = K.sum(y_true * y_pred, axis=axis)
		area_true = K.sum(y_true, axis=axis)
		area_pred = K.sum(y_pred, axis=axis)
		return ( (2*(intersection + smooth) / (area_true + area_pred + smooth)))
	    
	def hard_sorensen_dice(self,y_true, y_pred, axis=None, smooth=100):
		y_true_int = K.round(y_true)
		y_pred_int = K.round(y_pred)
		return self.soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

	def log_dice_loss_core(self,y_true, y_pred, axis=[1, 2],smooth=100):
		intersection = K.sum(y_true * y_pred, axis=axis)
		area_true = K.sum(y_true, axis=axis)
		area_pred = K.sum(y_pred,  axis=axis)
		batch_dice_coefs = K.log(2-( (intersection + smooth) / (area_true + area_pred + smooth)))
		return batch_dice_coefs

	def sorensen_dice_loss(self,y_true, y_pred, smooth=100):
		# Input tensors have shape (batch_size, height, width, classes)
		# User must input list of weights with length equal to number of classes
		#
		# Ex: for simple binary classification, with the 0th mask
		# corresponding to the background and the 1st mask corresponding
		# to the object of interest, we set weights = [0, 1]
		batch_dice_coefs = self.soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
		dice_coefs = K.mean(batch_dice_coefs, axis=0)
		w = K.constant(self.weights) / sum(self.weights)
		return (1 - K.sum(w * dice_coefs))

	def log_dice_loss(self,y_true, y_pred, axis=[1, 2],smooth=1):
		# Input tensors have shape (batch_size, height, width, classes)
		# User must input list of weights with length equal to number of classes
		#
		# Ex: for simple binary classification, with the 0th mask
		# corresponding to the background and the 1st mask corresponding
		# to the object of interest, we set weights = [0, 1]
		batch_dice_coefs = self.log_dice_loss_core(y_true, y_pred, axis=[1, 2])
		dice_coefs = K.mean(batch_dice_coefs, axis=0)
		w = K.constant(self.weights) / sum(self.weights)
		return K.sum(w * (1-dice_coefs))

	def dice_loss(self,y_true,y_pred):
		return self.sorensen_dice_loss(y_true,y_pred)

	def dice_cross_entropy(self,y_true,y_pred,l1=0.4,l2=0.6,smooth=100):
		loss_dice=self.log_dice_loss(y_true,y_pred)
		loss_entropy=self.weighted_categorical_crossentropy(y_true,y_pred)
		return l1*loss_dice+l2*loss_entropy

	def binary_dice_cross_entropy(self,y_true,y_pred,l1=0.5,l2=0.5,smooth=100):
		loss_dice=self.log_dice_loss(y_true,y_pred)
		loss_entropy= binary_crossentropy(y_true,y_pred)
		return l1*loss_dice+l2*loss_entropy

	def soft_jaccard(self,y_true, y_pred, axis=None, smooth=100):
		intersection = K.sum(y_true * y_pred, axis=axis)
		area_true = K.sum(y_true, axis=axis)
		area_pred = K.sum(y_pred, axis=axis)
		union = area_true + area_pred - intersection
		return (intersection + smooth) / (union + smooth) 

	def hard_jaccard(self,y_true, y_pred, axis=None, smooth=1):
		y_true_int = K.round(y_true)
		y_pred_int = K.round(y_pred)
		return self.soft_jaccard(y_true_int, y_pred_int, axis, smooth)


	def jaccard_loss(self,y_true, y_pred, smooth=100):
		intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
		sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
		jac = (intersection + smooth) / (sum_ - intersection + smooth)
		return (1 - jac) * smooth

	def weighted_categorical_crossentropy(self,y_true, y_pred):
		#weights = K.variable(np.array(self.weights))
		# scale predictions so that the class probas of each sample sum to 1
		#y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		#y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		#loss = y_true * K.log(y_pred) * weights
		#loss = -K.sum(loss, -1)
		#return loss

		 ndim = K.ndim(y_pred)
		 ncategory = K.int_shape(y_pred)[-1]
		 print(ndim,ncategory)
		# scale predictions so class probabilities of each pixel sum to 1
		 y_pred /= K.sum(y_pred, axis=(ndim-1), keepdims=True)
		 y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
		 w = K.constant(self.weights) * (ncategory / sum(self.weights))
		# first, average over all axis except classes
		 cross_entropies = K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim-1)))
		# print(y_pred)
		 return -K.sum(w * cross_entropies)

	def customized_loss(self,y_true, y_pred, alpha=0.0001, beta=3):
		"""
		linear combination of MSE and KL divergence.
		"""
		loss1 = losses.mean_absolute_error(y_true, y_pred)
		loss2 = losses.kullback_leibler_divergence(y_true, y_pred)
		#(alpha/2) *
		return  loss1 + beta * loss2

	def loss_Hausdorff_distance(self,y_true, y_pred):
		loss_loc = losses_distance.Weighted_Hausdorff_loss(y_true, y_pred)
		return loss_loc



###################################################Nifty losses##############################################################################

	def SS_DCE(self,y_true,y_pred):		
		loss_dice=self.dice_cross_entropy(y_true,y_pred)
		loss_sens=self.sensitivity(y_true,y_pred)
		loss_spec=self.specificity(y_true,y_pred) # self.sensitivity and self.specificity from evaluation file of auto
		loss_SS=2-loss_sens-loss_spec
		a=0.2
		b=0.8
		return a*loss_dice+b*loss_SS

	def SS_DL(self,y_true,y_pred):	
		loss_dice=self.dice_loss(y_true,y_pred)
		loss_sens=self.sensitivity(y_true,y_pred)
		loss_spec=self.specificity(y_true,y_pred) # self.sensitivity and self.specificity from evaluation file of auto
		loss_SS=2-loss_sens-loss_spec
		a=0.2
		b=0.8
		return a*loss_dice+b*loss_SS

	def SS_binary_accuracy(self,y_true,y_pred):	
		loss_b=binary_crossentropy(y_true,y_pred)
		loss_sens=self.sensitivity(y_true,y_pred)
		loss_spec=self.specificity(y_true,y_pred) # self.sensitivity and self.specificity from evaluation file of auto
		loss_SS=2-loss_sens-loss_spec
		a=0.5
		b=0.5
		return a*loss_b+b*loss_SS

	def SS_CE(self,y_true,y_pred):	
		loss_b=categorical_crossentropy(y_true,y_pred)
		loss_sens=self.sensitivity(y_true,y_pred)
		loss_spec=self.specificity(y_true,y_pred) # self.sensitivity and self.specificity from evaluation file of auto
		loss_SS=2-loss_sens-loss_spec
		a=0.4
		b=0.6
		return a*loss_b+b*loss_SS

	def specificity(self,y_pred, y_true):

		neg_y_true = 1 - y_true
		neg_y_pred = 1 - y_pred
		fp = K.sum(neg_y_true * y_pred)
		tn = K.sum(neg_y_true * neg_y_pred)
		specificity = tn / (tn + fp + K.epsilon())
		return specificity


	def sensitivity(self,y_pred, y_true):

		neg_y_pred = 1 - y_pred
		fn = K.sum(y_true * neg_y_pred)
		tp = K.sum(y_true * y_pred)
		sensitivity = tp / (tp + fn + K.epsilon())
		return sensitivity


###########################################CGAN losses ###########################################################################

	def loss_gan(self,y_true,y_pred,L1=10):
		g_loss_a2b = self.sce_criterion(y_true, tf.ones_like(y_true)) + L1 *self.abs_criterion(y_true, y_pred)+ L1*self.abs_criterion(y_pred,y_true)
		g_loss_b2a = self.sce_criterion(y_pred, tf.ones_like(y_pred)) + L1 *self.abs_criterion(y_pred, y_true)+ L1*self.abs_criterion(y_true, y_pred)
		g_loss = self.sce_criterion(y_pred, tf.ones_like(y_pred)) + self.sce_criterion(y_true, tf.ones_like(y_true)) +L1*self.abs_criterion(y_true, y_pred) + L1*self.abs_criterion(y_pred, y_true)
		g_total_loss=g_loss+g_loss_b2a+g_loss_a2b
		return g_total_loss


	def simple_loss_gan(self,y_true,y_pred,L1=10):
		g_loss= self.sce_criterion(y_true, tf.ones_like(y_pred)) + L1 *self.abs_criterion(y_true, y_pred) + self.mae_criterion(y_pred, tf.ones_like(y_true))
		return g_loss

	def abs_criterion(self,in_, target):
		return tf.reduce_mean(tf.abs(in_ - target))


	def mae_criterion(self,in_, target):
		return tf.reduce_mean((in_-target)**2)

	def sce_criterion(self,logits, labels):
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


