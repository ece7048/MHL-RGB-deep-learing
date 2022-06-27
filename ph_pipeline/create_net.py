#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk


from __future__ import division, print_function

import numpy as np
from tensorflow.keras.layers import Lambda, Input, Conv2D, Concatenate, MaxPooling2D, AveragePooling2D, AveragePooling1D, Dense, Flatten, Reshape, Activation, Dropout, Dense, MultiHeadAttention
from tensorflow.keras.models import Model
from ph_pipeline import regularization, config, main_net
from tensorflow.keras.applications import VGG16, VGG19, MobileNet, ResNet50, DenseNet121, DenseNet169, DenseNet201 #vgg16, vgg19, resnet_v1, resnet_v2, densenet, mobilenetv2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
import logging
import os
import os.path
from pypac import pac_context_for_url
import ssl

class create_net:

	def net(self, init1, init2, case, height, channels, classes, width,backbone_n="none",name='none'):
		model=[VGG16(include_top=False, weights='imagenet')]
		self.name=case
		self.name2=name
		self.main_model_names_transfer=[self.main_model]
		self.height=height
		self.width=width
		channels=channels
		if classes!=None:
			self.num_classes=classes
			print('redefine classes')
		context = ssl._create_unverified_context()
		with pac_context_for_url('https://www.google.com'): # put any website here
			multatt='off'
			if self.main_model[:3]=="RGB":
				multatt='on'
				#self.channels=1
				self.main_model=self.main_model[3:]
				if self.main_model=="":
					init_model=self.tune_RGB(height,width,2,backbone=backbone_n)
					print(backbone_n)
					model=[self.fine_tuned(init_model)]
					multatt='off'
			if self.main_model[:3]=="MHL":
				multatt='on'
				#self.channels=1
				self.main_model=self.main_model[3:]
				if self.main_model=="":
					init_model=self.tune_MHL(height,width,backbone=backbone_n)
					model=[self.fine_tuned(init_model)]
					multatt='off'
			if case!="":
				if channels==3:
					init_weights='imagenet'
					i_t=False
					pool=None
					cl=1000
				else:
					init_weights=None
					i_t=True
					pool=None
					cl=2000  #1024
			if self.main_model=="vgg16":
				init_model=VGG16(include_top=i_t, weights=init_weights,input_shape=(height, width,channels))
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="vgg19":
				init_model=VGG19(include_top=i_t, weights=init_weights,input_shape=(height, width,self.channels))
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="resnet50":
				init_model=ResNet50(include_top=i_t, weights=init_weights, input_shape=(height, width,channels)) #weights initial imagenet			
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="resnet101":
				init_model=ResNet101(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="resnet152":
				init_model=ResNet152(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="densenet121":
				init_model=DenseNet121(include_top=i_t, weights=init_weights, input_shape=(height, width,channels),pooling=pool,classes=cl) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="densenet169":
				init_model=DenseNet169(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="densenet201":
				init_model=DenseNet201(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=="mobilenet":
				init_model=MobileNet(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
				model=[self.fine_tuned(init_model)]
			elif self.main_model=='denseres171':
				model1='denres'
				model=self.transfer_learning(model1)
			elif self.main_model=='revgg66':
				model1="revgg66"  
				model=self.transfer_learning(model1)
			elif self.main_model=='DENRES':
				init_weights=np.array([str(self.store_model + "/weights_"+self.name+"_resnet50.h5"), str(self.store_model +  "/weights_"+self.name+"_densenet121.h5")])
				model=[self.denseres_171(i_t, init_weights, pool,cl, height, width)]
			elif self.main_model=="REVGG":
				init_weights=np.array([str(self.store_model +  "/weights_"+self.name + "_resnet50.h5"), str(self.store_model + "/weights_"+self.name + "_vgg16.h5")])	
				model=[self.revgg_66(i_t, init_weights, pool,cl, height, width)]
			elif self.main_model=='DENRESproto':
				model=[self.denres_proto()]
			else:			
				print('Error: no main model file')
			print(multatt)
			if multatt=='on':
				if self.main_model=="RGB":
					model=[self.rgb_attention(model[0],height,width,"RGB")]
				else:
					model=[self.rgb_attention(model[0],height,width,"MHL")]
		mod=model[0]
		if self.load_weights_main:
			logging.info("Load main weights from: {}".format(self.store_model+self.load_weights_main))
			mod.load_weights(self.store_model+self.load_weights_main)

		for p in (mod).layers:
			print(p.name.title(), p.input_shape, p.output_shape)


		return model		

	def __init__ (self,model) :
		args = config.parse_arguments()
		self.main_model=model
		self.load_weights_main=args.load_weights_main		
		self.store_model=args.store_txt
		self.batch_size=args.batch_size
		self.channels=args.channels
		self.num_classes=args.classes

        #preprocessing layer
	def rgb_attention(self,model2,height,width,channels=2,attention="RGB"):
		input=Input(shape=(height, width,3))
		model1=self.tuned_RGB(height,width,channels)
		model_file=str(self.store_model + "/weights_"+self.name+"_"+ attention+".h5")
		if os.path.exists(model_file):
			model1.load_weights(model_file,by_name=True, skip_mismatch=True)
		rg=model1(input)
		attrgb=model2(rg)	
		return Model(inputs=input, outputs=attrgb)

	def tune_MHL(self,height,width,backbone="none",attention=""):
		input=Input(shape=(height,width,3))
		s1=16
		s2=16
		c=1	
		if backbone=="none":
			Rc=Conv2D(1, (8, 8), padding="same", activation="relu")(input)
			print("case M-Head attention MHL ")
			s1=16
			s2=32
			c=1
		elif backbone=="resnet50":
			Rmodel=ResNet50(include_top=True, weights=None, input_shape=(height, width,3),pooling=None,classes=65536)
			model_file=str(self.store_model + "/weights_"+self.name2+"_resnet50"+ attention+".h5")
			if os.path.exists(model_file):
				Rmodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load weights of resnet')
			print(model_file)
			Rc=Rmodel(input)
			height=256
			width=256

			print("case M-Head attention resnet ")
		elif backbone=="densenet121":
			Dmodel=DenseNet121(include_top=True, weights=None, input_shape=(height, width,3),pooling=None,classes=65536)	
			model_file=str(self.store_model + "/weights_"+self.name2+"_densenet121"+ backbone+attention+".h5")

			if os.path.exists(model_file):
				Dmodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load denset weights')
			Rc=Dmodel(input)
			print("case M-Head attention densenet ")
			height=256
			width=256
		else:
			print("No none backbone network try resnet50, densenet121, or none!")
		Rd=Flatten(name='flatten_tunedR')(Rc)
		rgb=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Rd,Rd)
		#rgbr=Reshape([width,height,c])(rgb)
		#rgbm=Conv2D(1024, (8, 8), padding="same", activation="relu")(rgbr)
		#print(rgbm.shape)
		#rgbm=MaxPooling2D((8,8),strides=(s1,s2),padding="same")(rgbm)
		#rgbf=rgbm #Flatten(name='flat_rgb')(rgbm)
		#rg=Reshape([height,width,1])(rgbf)
		#new_DL=Dropout(0.3)(rg)
		#print(rgbf.shape,rgbm.shape)
		#rgbf=Flatten(name='flatten_t')(rgbr)
		#rgbd=Dense(height*width, activation="relu", name="dense4")(rgbf)
		rg=Reshape([height,width,1])(rgb)
		return Model(inputs=input, outputs=rg)



	def tune_RGB(self,height,width,channels=2,backbone="none",attention=""):
		input=Input(shape=(height,width,3))
		R=input[:,:,:,0]
		G=input[:,:,:,1]
		B=input[:,:,:,2]
		R=Reshape([height,width,1])(R)
		G=Reshape([height,width,1])(G)
		B=Reshape([height,width,1])(B)
		if backbone=="none":
			Rc=Conv2D(1, (8, 8), padding="same", activation="relu")(R)
			Gc=Conv2D(1, (8, 8), padding="same", activation="relu")(G)
			Bc=Conv2D(1, (8, 8), padding="same", activation="relu")(B)
			print("RGB")
			channels=2
		elif backbone=="resnet50":
			Rmodel=ResNet50(include_top=True, weights=None, input_shape=(height, width,1),pooling=None,classes=15000)
			model_file=str(self.store_model + "/weights_"+self.name2+"_"+ backbone+attention+".h5")
			if os.path.exists(model_file):
				Rmodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load weights of resnet')
			Rc=Rmodel(R)
			Gc=Rmodel(G)
			Bc=Rmodel(B)
			channels=3

			print("resnet!!")
		elif backbone=="densenet121":
			Dmodel=DenseNet121(include_top=True, weights=None, input_shape=(height, width,1),pooling=None,classes=15000)
			model_file=str(self.store_model + "/weights_"+self.name2+"_"+backbone+attention+".h5")
			if os.path.exists(model_file):
				Dmodel.load_weights(model_file,by_name=True, skip_mismatch=True)
				print('load weights of densenet')
			Rc=Dmodel(R)
			Gc=Dmodel(G)
			Bc=Dmodel(B)
			channels=3
			print("densenet!!")
			
		else:
			print("No none backbone network try resnet50, densenet121, or none!")

		Rd=Flatten(name='flatten_tunedR')(Rc)
		Gd=Flatten(name='flatten_tunedG')(Gc)
		Bd=Flatten(name='flatten_tunedB')(Bc)
		print(Rc.shape,Rd.shape)
		if channels==2:
			rgbs=MultiHeadAttention(num_heads=channels,key_dim=height,attention_axes=(1))(Rd,Gd)
			print("case M-Head attention: ",channels)
			rgb=Dense(65536, activation="relu", name="dense94")(rgbs)
		elif channels==3:
			rgb12=MultiHeadAttention(num_heads=channels,key_dim=height,attention_axes=(1))(Rd,Gd)
			rgb13=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Rd,Bd)
			rgb32=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Bd,Gd)
			rgb1=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Rd,Rd)
			rgb2=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Gd,Gd)
			rgb3=MultiHeadAttention(num_heads=2,key_dim=height,attention_axes=(1))(Bd,Bd)
			rgbC = Lambda(self.multi_concat, name="multi_concatenate" )([rgb12,rgb13,rgb32,rgb1,rgb2,rgb3])
			fl=Flatten(name='flatten_MHA')(rgbC)
			rgb=Dense(65536, activation="relu", name="dense65536")(fl)
			print("case M-Head attention: ",channels)
		else:
			rgb=Bd
			print("No multi-Head attention")
		#rgbr=Reshape([256,256,1])(rgb)
		#rgbm=Conv2D(1024, (8, 8), padding="same", activation="relu")(rgbr)
		#print(rgbm.shape)
		#rgbm=MaxPooling2D((8,8),strides=(8,16),padding="same")(rgbm)
		#rgbf=rgbm #Flatten(name='flat_rgb')(rgbm)
		#rg=Reshape([height,width,1])(rgbf)
                #new_DL=Dropout(0.3)(rg)
		#print(rgbf.shape,rgbm.shape)
		#rgbd=Dense(height, activation="relu", name="dense4")(rgbf)
		rg=Reshape([int(height/2),int(width/2),1])(rgb)
		return Model(inputs=input, outputs=rg)

	def fine_tuned(self,pretrained_model):
		for p in pretrained_model.layers:
			print(p.name.title(), p.input_shape, p.output_shape)
		new_DL=pretrained_model.output
		new_DL=Flatten(name='flatten_tuned')(new_DL)
		new_DL=Dense(1024, activation="relu", name='dense2')(new_DL)	#64
		new_DL=Dropout(0.3,name='drop2')(new_DL)
		new_DL=Dense(512, activation="relu",name='dense3')(new_DL)	#64
		new_DL=Dropout(0.3,name='drop4')(new_DL)	
		new_DL=Dense(self.num_classes, activation="softmax",name='dense45')(new_DL) #2
		#new_DL=Reshape([2,2,1])(new_DL)
		#print(new_DL)
		return Model(inputs=pretrained_model.input, outputs=new_DL)

	def transfer_learning(self,model_n):
		init_weights='imagenet'
		i_t=False
		pool=None
		cl=1000
		height=self.height
		width=self.width

		model_init=VGG16(include_top=False, weights='imagenet')
		model=[model_init]
		if model_n=='denres':
			init_model1=DenseNet121(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
			model1=self.fine_tuned(init_model1)
			init_model=ResNet50(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet                 
			model2=self.fine_tuned(init_model)
			#init_weights=np.array([str(self.store_model + "/weights_"+self.name+"_resnet50.h5"), str(self.store_model +  "/weights_"+self.name+"_densenet121.h5")])
			#model3=self.denseres_171(i_t, init_weights, pool,cl, height, width)
			model=[model1,model2]
			self.main_model_names_transfer=["resnet50","densenet121"]
		elif model_n=='revgg':
			init_model1=VGG16(include_top=i_t, weights=init_weights,input_shape=(height, width,self.channels)) #weights initial imagenet
			model1=self.fine_tuned(init_model1)
			init_model=ResNet50(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet                 
			model2=self.fine_tuned(init_model)
			#init_weights=np.array([str(self.store_model +  "/weights_"+self.name + "_resnet50.h5"), str(self.store_model + "/weights_"+self.name + "_vgg16.h5")])
			#model3=self.revgg_66(i_t, init_weights, pool,cl, height, width)
			model=[model1,model2]
			self.main_model_names_transfer=["resnet50","vgg16"]
		else:
			print('notransfer')
		return model



	def revgg_66(self,i_t, init_weights, pool,cl,height, width):
		input=Input(shape=(height, width,self.channels))
		res=ResNet50(include_top=i_t, weights='imagenet', input_shape=(height,width,self.channels))
		res=self.fine_tuned(res)
		vgg=VGG16(include_top=i_t, weights='imagenet',input_shape=(height,width,self.channels))
		vgg=self.fine_tuned(vgg)
		print('res: ')
		res.load_weights(init_weights[0])
		vgg.load_weights(init_weights[1])
		for p in res.layers:
			print(p.name.title(), p.input_shape, p.output_shape)

		res112=res.get_layer("conv1_relu").output #frame 64 tensorflow=2.2.0  activation_1, tensorflow>=2.3.0 conv1_relu

		res28c=res.get_layer("conv3_block2_2_relu").output #128 tensorf=2.2.0 activation_12,  tensorflow>=2.3.0 conv3_block2_2_relu
		res28a=res28c #AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res28c)

		res28b=res.get_layer("conv2_block3_out").output #256 tensorflow=2.2.0  activation_9   tensorflow>=2.3.0  conv2_block3_out

		res28=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res28b)
		res14b=res.get_layer("conv3_block4_out").output #512 tensorflow==2.2.0 activation_21 tensorflow>=2.3.0 conv3_block4_out

		res14=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res14b)

		res28b=res.get_layer("conv2_block3_out").output #256 tensorflow=2.2.0  activation_9   tensorflow>=2.3.0  conv2_block3_out

		res28=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res28b)
		res14b=res.get_layer("conv3_block4_out").output #512 tensorflow==2.2.0 activation_21 tensorflow>=2.3.0 conv3_block4_out

		res14=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res14b)
		res7=res.get_layer("conv5_block3_2_relu").output #512 tensorflo==2.2.0  activation_47 tensorflow>=2.3.0 conv5_block3_2_relu

		vgg112=vgg.get_layer("block1_pool").output #64
		vgg56=vgg.get_layer("block2_pool").output
		vgg28a=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(vgg56) #128
		vgg28=vgg.get_layer("block3_pool").output #256
		vgg14=vgg.get_layer("block4_pool").output #512
		vgg7=vgg.get_layer("block5_pool").output #512

		reso=Model(inputs=res.input,outputs=[res112,res28a,res28,res14,res7])
		vggo=Model(inputs=vgg.input,outputs=[vgg112,vgg28a,vgg28,vgg14,vgg7])

		resm=reso(input)
		vggm=vggo(input)

		concatenate_layer112 = Lambda(self.concat, name="concatenate1" )([resm[0], vggm[0]])
		concatenate_layer28a = Lambda(self.concat, name="concatenate2" )([resm[1], vggm[1]])
		concatenate_layer28 = Lambda(self.concat, name="concatenate3" )([resm[2], vggm[2]])
		concatenate_layer14 = Lambda(self.concat, name="concatenate4" )([resm[3], vggm[3]])
		concatenate_layer7 = Lambda(self.concat, name="concatenate5" )([resm[4], vggm[4]])

		c11=Conv2D(512, (3, 3), padding="same", activation="relu")(concatenate_layer112)
		c21=Conv2D(1024, (3, 3), padding="same", activation="relu")(concatenate_layer28a)
		c31=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer28)
		a312=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c31)
		a31=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(a312)

		c4=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer14)
		a44=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c4)
		a4=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(a44)

		a111=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c11)
		a112=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(a111)
		a11=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(a112)

		c12=Conv2D(1024, (2, 2), padding="same", activation="relu")(a11)
		a12=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c12) #stride (2,2) if the tensorflow>2.2.0 else (1,1)

		c13=Conv2D(2048, (2, 2), padding="same", activation="relu")(a12)
		a13=AveragePooling2D((1, 1), strides = (1, 1), padding = "same")(c13)

		a21=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c21)
		c22=Conv2D(2048, (2, 2), padding="same", activation="relu")(a21)
		a22=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c22)
		c5=Conv2D(2048, (1, 1), padding="same", activation="relu")(concatenate_layer7)


		concatenate_f1 = Lambda(self.concat, name="concatenatef1" )([a13,a22])
		concatenate_f2 = Lambda(self.concat, name="concatenatef2" )([a31,a4])
		concatenate_f3 = Lambda(self.concat, name="concatenatef3" )([concatenate_f1, concatenate_f2])
		concatenate_f4 = Lambda(self.concat, name="concatenatef4" )([concatenate_f3, c5])


		DL = concatenate_f4
		new_DL=Flatten(name="f1")(DL)
		new_DL=Dense(12096, activation="relu", name="dense1")(new_DL)
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(2048, activation="relu", name="dense2")(new_DL)
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(512, activation="relu", name="dense3")(new_DL)    #64
		new_DL=Dropout(0.3)(new_DL)
		new_DLf=Dense(self.num_classes, activation="sigmoid", name="dense4")(new_DL) #2

		return Model(inputs=input, outputs=new_DLf)



	def revgg_69(self,i_t, init_weights, pool,cl,height, width):
		input=Input(shape=(height, width,self.channels))
		res=ResNet50(include_top=i_t, weights='imagenet', input_shape=(height, width,self.channels))
		res=self.fine_tuned(res)
		vgg=VGG19(include_top=i_t, weights='imagenet',input_shape=(height, width,self.channels))
		vgg=self.fine_tuned(vgg)
		print('res: ')
		res.load_weights(init_weights[0])
		vgg.load_weights(init_weights[1])
		for p in res.layers:
			print(p.name.title(), p.input_shape, p.output_shape)

		res112=res.get_layer("conv1_relu").output #frame 64 activa 1
		res28c=res.get_layer("conv3_block2_2_relu").output #128 activ12
		res28a=res28c #AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res28c)
		res28b=res.get_layer("conv2_block3_out").output #256 activ10
		res28=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res28b)
		res14b=res.get_layer("conv3_block4_out").output #512 activation_22
		res14=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(res14b)
		res7=res.get_layer("conv5_block3_2_relu").output #512 activ_48

		vgg112=vgg.get_layer("block1_pool").output #64
		vgg56=vgg.get_layer("block2_pool").output
		vgg28a=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(vgg56) #128
		vgg28=vgg.get_layer("block3_pool").output #256
		vgg14=vgg.get_layer("block4_pool").output #512
		vgg7=vgg.get_layer("block5_pool").output #512

		reso=Model(inputs=res.input,outputs=[res112,res28a,res28,res14,res7])
		vggo=Model(inputs=vgg.input,outputs=[vgg112,vgg28a,vgg28,vgg14,vgg7])

		resm=reso(input)
		vggm=vggo(input)

		concatenate_layer112 = Lambda(self.concat, name="concatenate1" )([resm[0], vggm[0]])
		concatenate_layer28a = Lambda(self.concat, name="concatenate2" )([resm[1], vggm[1]])
		concatenate_layer28 = Lambda(self.concat, name="concatenate3" )([resm[2], vggm[2]])
		concatenate_layer14 = Lambda(self.concat, name="concatenate4" )([resm[3], vggm[3]])
		concatenate_layer7 = Lambda(self.concat, name="concatenate5" )([resm[4], vggm[4]])


		c11=Conv2D(512, (3, 3), padding="same", activation="relu")(concatenate_layer112)
		c21=Conv2D(1024, (3, 3), padding="same", activation="relu")(concatenate_layer28a)
		c31=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer28)
		a31=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c31)

		c4=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer14)
		a4=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(c4)


		a111=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c11)
		a11=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(a111)
		c12=Conv2D(1024, (2, 2), padding="same", activation="relu")(a11)
		a12=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c21)
		c13=Conv2D(2048, (2, 2), padding="same", activation="relu")(a12)
		a13=AveragePooling2D((1, 1), strides = (1, 1), padding = "same")(c13)
		#a13=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c13)

		a21=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c2)
		c22=Conv2D(2048, (2, 2), padding="same", activation="relu")(a21)
		a22=MaxPooling2D((2, 2), strides = (2, 2), padding = "same")(c22)
		c5=Conv2D(2048, (1, 1), padding="same", activation="relu")(concatenate_layer7)
		
		
		concatenate_f1 = Lambda(self.concat, name="concatenatef1" )([a13, a22])
		concatenate_f2 = Lambda(self.concat, name="concatenatef2" )([a31, a4])
		concatenate_f3 = Lambda(self.concat, name="concatenatef3" )([concatenate_f1, concatenate_f2])
		concatenete_f4 = Lambda(self.concat, name="concatenatef3" )([concatenate_f3, c5])


		DL = concatenate_f4
		new_DL=Flatten(name="f1")(DL)
		new_DL=Dense(50096, activation="relu", name="dense1")(new_DL)
		new_DL1=Dropout(0.3)(new_DL)
		new_DL=Dense(1024, activation="relu", name="dense3")(new_DL1)
		new_DL00=Dense(512, activation="relu", name="dense6")(new_DL)    #64
		new_DL01=Dropout(0.3)(new_DL00)
		new_DLf=Dense(self.num_classes, activation="sigmoid", name="dense8")(new_DL01) #2

		return Model(inputs=input, outputs=new_DLf)


	def denres_proto(self):
		init_weights='imagenet'
		i_t=False
		pool=None
		cl=1000
		init_weights=np.array([str(self.store_model + "/weights_"+self.name+"_resnet50.h5"), str(self.store_model +  "/weights_"+self.name+"_densenet121.h5")])
		model=[self.denseres_171(i_t, init_weights, pool, cl, self.height, self.width)]
		proto=denres_171(i_t=i_t, init_weights=[init_weights[0],init_weights[1]], input_shape=(self.height, self.width, self.channels))
		proto.load_weights(str(self.store_model + "/weights_"+self.name+"_denseres171.h5"))
		feature=proto.get_layer("f2").output
		print('proto case')
		return Model(inputs=proto.input, outputs=[feature])





	def denseres_171(self,i_t, init_weights, pool,cl,height, width):

		input=Input(shape=(height, width,self.channels))
		res=ResNet50(include_top=i_t, weights=None, input_shape=(height, width,self.channels))
		res=self.fine_tuned(res)
		dense=DenseNet121(include_top=i_t, weights=None, input_shape=(height, width,self.channels), pooling=pool,classes=cl)
		dense=self.fine_tuned(dense)
		print('res: ')
		if os.path.exists(init_weights[0]):
			res.load_weights(init_weights[0],by_name=True, skip_mismatch=True)
		if os.path.exists(init_weights[1]):
			dense.load_weights(init_weights[1],by_name=True, skip_mismatch=True)
		for p in res.layers:
			print(p.name.title(), p.input_shape, p.output_shape)


		res56=res.get_layer("conv2_block3_out").output
#		98, 89,71,59
#		49,40,22,10
		dense56=dense.get_layer("pool2_relu").output
		res28=res.get_layer("conv3_block4_out").output
		dense28=dense.get_layer("pool3_relu").output
		res14=res.get_layer("conv4_block6_out").output
		dense14=dense.get_layer("pool4_relu").output
		res7=res.get_layer("conv5_block3_out").output
		dense7=dense.get_layer("relu").output

		reso=Model(inputs=res.input,outputs=[res56,res28,res14,res7])
		denseo=Model(inputs=dense.input,outputs=[dense56,dense28,dense14,dense7])

		resm=reso(input)
		densem=denseo(input)

		concatenate_layer56 = Lambda(self.concat, name="concatenate1" )([resm[0], densem[0]])
		concatenate_layer28 = Lambda(self.concat, name="concatenate2" )([resm[1], densem[1]])
		concatenate_layer14 = Lambda(self.concat, name="concatenate3" )([resm[2], densem[2]])
		concatenate_layer7 = Lambda(self.concat, name="concatenate4" )([resm[3], densem[3]])


		c11=Conv2D(512, (3, 3), padding="same", activation="relu")(concatenate_layer56)	
		c21=Conv2D(1024, (3, 3), padding="same", activation="relu")(concatenate_layer28)

		c31=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer14)
		a31=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c31)

		c4=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer7)
		a4=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(c4)


		a11=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c11)
		c12=Conv2D(1024, (2, 2), padding="same", activation="relu")(a11)
		a12=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c12)
		c13=Conv2D(2048, (2, 2), padding="same", activation="relu")(a12)
		a13=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c13)

		a21=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c21)
		c22=Conv2D(2048, (2, 2), padding="same", activation="relu")(a21)
		a22=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c22)

		concatenate_f1 = Lambda(self.concat, name="concatenatef1" )([a13, a22])
		concatenate_f2 = Lambda(self.concat, name="concatenatef2" )([a31, a4])
		concatenate_f3 = Lambda(self.concat, name="concatenatef3" )([concatenate_f1, concatenate_f2])

		DL = concatenate_f3
		new_DL=Flatten(name="f2")(DL)
		new_DL=Dense(1024, activation="relu", name="dense1")(new_DL)   #64
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(512, activation="relu", name="dense2")(new_DL)    #64
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(self.num_classes, activation="sigmoid", name="dense3")(new_DL) #2
                
		return Model(inputs=input, outputs=new_DL)

	def concat(self,input_list):
		height=input_list[0]
		weight=input_list[1]
		x=Concatenate(axis=-1)([height,weight])
		return x
	def multi_concat(self,input_list):
		y=input_list[0]
		for i in range(2,len(input_list)):
			height=input_list[i-1]
			weight=input_list[(i)]
			x=Concatenate(axis=-1)([height,weight])
			y=Concatenate(axis=-1)([y,x])	
		print("the multi_concat shape is : ",y.shape)
		return y
