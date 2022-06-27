import skimage
import skimage.io
import skimage.transform
import numpy as np


import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.cm as cm
from ph_pipeline import config, create_net, run_model
from tensorflow.keras.models import model_from_json
#from tensorflow import placeholder
import tensorflow.compat.v1 as tf1
#tf1.disable_v2_behavior()
from tensorflow.keras import backend as K
import tensorflow as tf
from skimage import io
from scipy import ndimage,misc
from skimage.transform import resize
from tensorflow.python.framework import ops
from scipy.stats import gmean
from tensorflow.python.ops import gen_nn_ops
#tf1.compat.v1.disable_eager_execution()
import glob
import cv2

# synset = [l.strip() for l in open('synset.txt').readlines()]
def image_preprocess(resized_inputs,channel=3):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    channel_means = tf1.constant([123.68, 116.779, 103.939],
        dtype=tf1.float32, shape=[1, 1, 1, channel], name='img_mean')
    return resized_inputs - channel_means


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image2(path, normalize=False,x=224,y=224,c=1):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    image_list=[]
    image_list = sorted(glob.glob(path +"/*"))
    total=np.array(image_list).shape[0]
    img=[]
    total_img=[]
    for i in range(total):
        img.append(skimage.io.imread(image_list[i]))
    img=np.array(img)
    
    for o in range(total):
        if normalize:
            img1 = (img[o] / (np.max(img[o])))
            assert (0 <= img1).all() and (img1 <= 1.0).all()
        else:
            img1=img[o]
    # print "Original Image Shape: ", img.shape
    # we crop image from center
        short_edge = min(img1.shape[:2])
        yy = int((img1.shape[0] - short_edge) / 2)
        xx = int((img1.shape[1] - short_edge) / 2)
        crop_img = img1[yy: yy + short_edge, xx: xx + short_edge,:]
        # resize to 224, 224
        #print(np.array(crop_img).shape)
        resized_img = skimage.transform.resize(crop_img, (x, y,c), preserve_range=True) # do not normalize at transform. 
        total_img.append(resized_img)
    return np.array(total_img)

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1



def visualize2(image, conv_output, conv_grad, gb_vizi,store,x=224,y=224,num_chan=1):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad         
    img=255*image
    total=np.array(img).shape[0]
    print("grad_val:",grads_val.shape)
    if (len(grads_val.shape)==4):
        weights = np.mean(grads_val, axis = (1, 2)) # alpha_k, [512]
        c = np.zeros([output.shape[0],output.shape[1],output.shape[2]], dtype = np.float32) # [7,7]
    else:
        weights = (grads_val) # alpha_k, [512]
        c = np.zeros([output.shape[0],output.shape[1]], dtype = np.float32) # [7,7]
    print("weights: ", weights.shape)
    print("output: ",output.shape)
    print("cam: ",c.shape)   
    cam=[]
    # Taking a weighted average
    for o in range(total):
        cam1=c[o]
        w_o=weights[o,:]
        for i, w in enumerate(w_o):
            if (len(grads_val.shape)==4):
                cam1 += w * output[o,:, :, i]
            else:
                cam1 += w * output[o, i]
            #print("w is : ",w)
            #print("output ; ", output[o,:5,:5,i], i)
        cam1=np.abs(cam1)
        camax = np.maximum(cam1, 0)
        camt = camax / np.max(camax)
        cam.append(camt)
        print("MAX cam is: ", np.max(camax), np.max(camt))

    cam=np.array(cam)#, dtype = np.uint8)
    print("cam shape: ", cam.shape)
    cam = resize(cam, (total,x,y))#, preserve_range=True)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * c)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_ar=[]
    gb1_ar=[]
    im_ar=[]
    ch_at=[]
    # Create an image with RGB colorized heatmap
    for h in range(total):
    # Superimpose the heatmap on original image
        im_a=tf1.keras.preprocessing.image.array_to_img(np.array(img[h]))
        cam_o=(255*cam[h]).astype(np.uint8)
        ch = cv2.applyColorMap(cam_o, cv2.COLORMAP_JET)
        ch= cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
        s_im = np.array(ch[h]) * 0.4 + np.array(img[h])
        jet_ar =s_im  #tf1.keras.preprocessing.image.array_to_img(s_im)
        ch_at.append(ch)
        im_ar.append(im_a)
    s_imgh = jet_ar
    img2=im_ar
    ch2=ch_at
    print("Max cam : ",np.max(ch2),ch2[10])
    #s_img=tf1.keras.preprocessing.image.array_to_img(s_imgh)
    s_img=s_imgh

    gb_viz=gb_vizi
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
    #gd_gb=cam
    print("gp shape: ",np.array(gb_vizi).shape)

    if len(gb_viz.shape)==4:
        gd_gb = np.dstack((
           gb_viz[:,:,:,0]* cam ,
            gb_viz[:,:, :, 1] * cam,
           gb_viz[:,:, :, 2] * cam,
        ))
        gd_gb=[]
        gd_gb.append(gb_viz[:,:,:,0]*cam)
        gd_gb.append(gb_viz[:,:,:,1]*cam)
        gd_gb.append(gb_viz[:,:,:,2]*cam)
    #gb0=np.reshape(gd_gb,[total,gd_gb.shape[1],gd_gb.shape[1]],3)
    #img1=np.reshape(img,[total,img.shape[1],img.shape[1],num_chan])
    #gb1=np.array(gb0)*0.8 +np.array(img[h])
    #gd_gb== 255 * gd_gb / np.max(gd_gb)
    #gbw1=np.reshape(gb1,[gd_gb.shape[1],gd_gb.shape[2],gd_gb.shape[0]])
        gb0=np.array(gd_gb)
        gb0=np.reshape(gb0,[gb0.shape[1],gb0.shape[2],gb0.shape[3],3])
    #for h in range(total):
     #      gb11=(255*gb0[h])*0.7 +np.array(img[h])
      #     gb_ar=tf1.keras.preprocessing.image.array_to_img(gb11)
       #    gb1_ar.append(gb_ar)
    #gb1=gb1_ar
    else:
        gb0=np.array(gb_viz)
        
    for i in range(total):         
           fig = plt.figure()    
           ax = fig.add_subplot(141)
           imgplot = plt.imshow(img2[i])
           ax.set_title('Input Image')
           #fig = plt.figure(figsize=(12, 16))    
           ax = fig.add_subplot(142)
           imgplot = plt.imshow(ch2[i])
           ax.set_title('Grad-CAM')    
           ax = fig.add_subplot(143)
           imgplot = plt.imshow(s_img[i])
           ax.set_title('GCAM+image')
           ax = fig.add_subplot(144)
           imgplot = plt.imshow(gb0[i])
           ax.set_title('guided GCAM')
           plt.savefig(store+'/image_'+str(i)+'.png')

    return np.array(ch2)


@ops.RegisterGradient("GuidedRelu2")
def _GuidedReluGrad(op, grad):
    return tf1.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf1.zeros_like(grad))


#from slim.nets import resnet_v1
#slim = tf.contrib.slim

def layers_print2(model,inputs):
     activations=[]
     inp = model.input                              # input placeholder
     outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
     functors = [K.function(inp, [out]) for out in outputs]   # evaluation functions
     # Testin
     list_inputs = inputs
    # layer_outputs = [func(list_inputs)[0] for func in functors]
     #for layer_activations in layer_outputs:
      #   activations.append(layer_activations)

     return outputs,functors  #,activations


def viewer2(X,model,store, point=[7,9],model_name='none',height=256,width=256,channels=1,classes=1,case='test',batch=1,label=[0,0,0,1],rot=0,backbone='none',input_n=1):

    #input one class per time to create the pca map
    # all the images same class in other case wrong

    tf1.compat.v1.disable_eager_execution()
    img1 = load_image2(X,x=height,y=width,c=channels) #"./demo.png", normalize=False)
    img2=load_image2(X,normalize=True,x=height,y=width,c=channels)
    batch_size=np.array(img1).shape[0]
    img1=ndimage.rotate(img1,rot,reshape=False)
    img2=ndimage.rotate(img2,rot,reshape=False)
    print(img1.shape)
    blabel=[]
    batch1_img_all = img1.reshape((batch_size, height, width, channels))
    batch2_img_all = img2.reshape((batch_size, height, width, channels))
    batch1_img=batch1_img_all
    batch2_img=batch2_img_all
    #batch_size=1
    batch1_labe = np.array(label,dtype=np.float32)  # 1-hot result for Boxer
    batch1_labe = batch1_labe.reshape(1, -1)
    for o in range(batch_size):
        blabel.append(batch1_labe)
    batch1_label=np.array(blabel)
    imageK=K.placeholder(shape=(batch_size,height,width,channels))
    label = K.placeholder(shape=(batch_size, classes))
    if input_n>=2:
        batch1_label=np.array(blabel)
        while batch1_img.shape[3]<(input_n):
            batch1_img=np.concatenate((batch1_img,batch1_img_all),axis=3)

    cn=create_net.create_net(model_name)
    print("The model name is: ",model_name)
    model_structure=cn.net([], [], case,height,channels,(classes),width,backbone) # classes +1 to take into account the background
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    model_str=model_structure[0]
    model_str.load_weights(file_store)
    end_p ,functors= layers_print2(model_str,batch1_img)
    end_points=end_p
    print(len(end_points))
    deliver=end_p
    prob =(deliver[(len(end_points)-point[0])])  #,dtype=tf1.float32) # after softmax
    #prob_im=image_feature[len(end_points)-point[0]]
    cost = (-1) * tf1.reduce_sum(tf1.multiply(batch1_label, tf1.log(prob)), axis=1) #one image per time in other case need modification 
    target_conv_layer =deliver[len(end_points)-point[1]]
    #tg=image_feature[len(end_points)-point[1]]
    net=model_str.output
    y_c = tf1.reduce_sum(tf1.multiply(net, batch1_label), axis=1)#net
    target_conv_layer_grad = K.gradients(y_c, target_conv_layer)[0]

    # Guided backpropagtion back to input layer
    gb_grad = K.gradients(cost[1], model_str.input)[0] #[0] before
    iterate=K.function([model_str.input,label],[cost,gb_grad,prob])
    iterate2=K.function([model_str.input,label],[y_c,target_conv_layer_grad,target_conv_layer])
    #[target_conv_layer,model_str.input]
    print("prob : ", prob)
    #print("cost : ",cost)
    cost_np,gb_grad_value, prob_np=iterate([batch1_img,batch1_label])
    y_c_np,target_conv_layer_grad_value,tg=iterate2([batch1_img,batch1_label])#([tg,batch1_img])
    target_conv_layer_value=tg
    #print("prob after: ", prob_np)
    #print("prob image_active: ", prob_im)
    #print("Y_c ",y_c)     
    #print("Y_C result= ",y_c_np)
    y_pred = model_str.predict_generator(batch1_img)
    #print("prediction= ",y_pred)  
    #print("CCN  grad out: ",target_conv_layer_grad_value)
    print("CNN before: ", target_conv_layer)
    #print("CNN after: ",target_conv_layer_value)
    #print("CNN image: ",tg)
    imag=visualize2(batch1_img, target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value,store,x=height,y=width,num_chan=channels)
    gmean(imag,batch1_img,store)


def gmean(X,Y,store):

    r=np.array(X/255)
    g=np.array(Y/255)
    imn2=gaverage(r)
    imn=gaverage(g)
    im2=(255*imn2).astype(np.uint8)
    im=(255*imn).astype(np.uint8)
    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    #im= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2= cv2.applyColorMap(im2, cv2.COLORMAP_JET)
    #im2= cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    #im=np.reshape(im,[r.shape[1],r.shape[2],r.shape[3]])
    #im2=np.reshape(im2,[r.shape[1],r.shape[2],r.shape[3]])
    im1=im   #tf1.keras.preprocessing.image.array_to_img(im_m)
    plt.figure(figsize=[7.3, 7.3])
    fig = plt.figure()
    ax = fig.add_subplot(121)
    imgplot = plt.imshow(im2)
    ax.set_title('Average Image ')
    ax = fig.add_subplot(122)
    imgplot = plt.imshow(im1)
    ax.set_title('Average GRANCAM ')
    plt.savefig(store+'/image_Average_Gmean.png')


def gaverage(X):
    lenf=X.shape[3]
    aver=np.zeros([1,X.shape[1],X.shape[2],X.shape[3]])
    for o in range (X.shape[0]):
        aver=aver+X[o]/lenf
    average=np.reshape(aver,[X.shape[1],X.shape[2],X.shape[3]])
    return average    


def pca(X,Y,store):

    r=np.array(X/255)
    g=np.array(Y/255)
    #b=np.array(X[:,:,:,2])
    dim=r.shape[0]
    feat=r.shape[1]*r.shape[2]*r.shape[3]
    r1=np.reshape(r,[dim,feat])
    g1=np.reshape(g,[dim,feat])
    #b1=np.reshape(b,[dim,feat])
    print(np.array(r).shape)
    pca_r = PCA(n_components=40)
    pca_r_trans= pca_r.fit_transform(r1)
    pca_g = PCA(n_components=40)
    pca_g_trans = pca_g.fit_transform(g1)
    #pca_b = PCA(n_components=40)
    #pca_b_trans = pca_b.fit_transform(b1)

    pca_r_org = pca_r.inverse_transform(pca_r_trans)
    pca_g_org = pca_g.inverse_transform(pca_g_trans)
    #pca_b_org = pca_b.inverse_transform(pca_b_trans)
    imn2=np.array(pca_g_org)
    imn=np.array(pca_r_org)
    im2=(255*imn2).astype(np.uint8)
    im=(255*imn).astype(np.uint8)
    im=np.reshape(im,[dim,r.shape[1],r.shape[2],r.shape[3]])
    im2=np.reshape(im2,[dim,r.shape[1],r.shape[2],r.shape[3]])
    for o in range(39):
        im_m=im[o] #0 median
        im1=im_m   #tf1.keras.preprocessing.image.array_to_img(im_m)
        plt.figure(figsize=[7.3, 7.3])
        fig = plt.figure()
        ax = fig.add_subplot(121)
        imgplot = plt.imshow(im2[o])
        ax.set_title('PCA Image component '+str(o))
        ax = fig.add_subplot(122)
        imgplot = plt.imshow(im1)
        ax.set_title('PCA GRANCAM component '+str(o))
        plt.savefig(store+'/image_PCA_compos_'+str(o)+'.png')
