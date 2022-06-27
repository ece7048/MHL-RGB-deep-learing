#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk
from __future__ import division, print_function
import argparse
from ph_pipeline import config, create_net, run_model
from tensorflow.keras.models import model_from_json
from ph_pipeline import  regularization 
import numpy as np
from sklearn.metrics import roc_curve, auc
from ph_pipeline.regularization import data_augmentation

class store_model:

	def __init__ (self) :
		args = config.parse_arguments()
		self.store_model_path=args.store_txt
		self.classes=args.classes
		self.loss_main=args.loss_main
		#self.loss_roi=args.loss_roi
		self.width=args.width
		self.height=args.height
		self.model_name=args.main_model	
		self.type_analysis=args.type_analysis


	def set_model(self,name_model_json, name_model_h5, model_weights):

		# serialize model to JSON
		model_json = model_weights.to_json()
		with open(self.store_model_path + name_model_json, "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model_weights.save_weights(self.store_model_path + name_model_h5)
		print("Saved model to disk")


	def get_model(self,name_model_json, name_model_h5):
		# load json and create model
		json_file = open(self.store_model_path + name_model_json, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(self.store_model_path + name_model_h5)
		print("Loaded model from disk")
		return loaded_model

	def load_best_callback(self,model_structure,X,Y,case, weight_name):

		#Xaug, Yaug= [], []
		#Xaug=X
		#Yaug=Y
		#for i in data_augmentation(X,Y):
		#	Xaug.append(i[0])
		#	Yaug.append(i[1])
		image, mask= (X) , (Y)
		print(X.shape,Y.shape)
		if self.type_analysis=='SE':
			_, height, width, channels = image.shape					
			_, classes, _ ,_ = mask.shape
		else:
			_, height, width, channels = image.shape
			_, classes = mask.shape

		cn=create_net.create_net(case)
		print('load weights')
		#model_structure.load_weights(self.store_model_path + '/weights_%s_%s.h5' %(weight_name,case))
		print("Load model from: ")

		file_store=(self.store_model_path + '/weights_%s_%s.h5' %(weight_name,case))#hdf5
		print(file_store)
		model_structure.load_weights(file_store)
		rm=run_model.run_model(case)

		fpr,tpr,aucp=dict(),dict(),dict()
		thresholds=0
		if self.type_analysis=='SE':
			pred, metric=rm.run_testing(self.loss_main,model_structure, X, Y)				
			return pred, metric
		else:
			y_pred_keras = model_structure.predict(X) #.ravel()
			print(np.array(y_pred_keras.shape))
			print('before the discretization')
			#print((y_pred_keras[4,:]))
			y_pred=np.array(y_pred_keras)
			y_pred_keras=np.array(y_pred_keras)
			y_pred_keras=np.absolute(np.array(y_pred_keras)/np.max(y_pred_keras))
			y_pred_keras[np.arange(len(y_pred_keras)), y_pred_keras.argmax(1)] = 1
			print((y_pred_keras[4,:],np.array(y_pred_keras).shape[0]))
			length=np.array(y_pred_keras).shape[0]
			for i in range(length):
				y_max=np.max(y_pred_keras[i,:])
				y_pred_keras[i]=np.where(y_pred_keras[i]==y_max, 1,0 ) #change to predict continious
			
			print('after:')
			print((y_pred_keras[4,:]))
			for i in range(np.array(y_pred_keras.shape[1])):
				fpr[i], tpr[i], thresholds = roc_curve(Y[:,i], y_pred[:,i])
				aucp[i] = auc(fpr[i], tpr[i])
			return fpr, tpr, aucp, y_pred_keras
