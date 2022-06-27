import skimage
import skimage.io
import skimage.transform
import numpy as np

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
from tensorflow.python.ops import gen_nn_ops
#tf1.compat.v1.disable_eager_execution()

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
def load_image(path, normalize=True,x=224,y=224,c=1):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    img = skimage.io.imread(path)
    if normalize:
        img = (img / (np.max(img)))
        assert (0 <= img).all() and (img <= 1.0).all()


    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    print(np.array(crop_img).shape)
    resized_img = skimage.transform.resize(crop_img, (x, y,c), preserve_range=True) # do not normalize at transform. 
    return resized_img

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



def visualize(image, conv_output, conv_grad, gb_vizi,store,x=224,y=224):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad          
    img=255*image
    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    print("weights: ", weights.shape)
    print("output: ",output.shape)
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)	# [7,7]
    print("cam: ",cam.shape)   

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    print("cam2: ",cam)
    cam1 = np.maximum(cam, 0)
    cam = cam1 / np.max(cam1) # scale 0 to 1.0
    cam = resize(cam, (x,y), preserve_range=True)

    print("MAX cam is: ", np.max(cam1), np.max(cam))


    #img = image.astype(float)
    #img -= np.min(img)
    #img /= img.max()
    # print(img)
    ch = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    ch= cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * cam)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf1.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf1.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    s_img = tf1.keras.preprocessing.image.array_to_img(superimposed_img)

           
    fig = plt.figure()    
    ax = fig.add_subplot(141)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')
    
    fig = plt.figure(figsize=(12, 16))    
    ax = fig.add_subplot(142)
    imgplot = plt.imshow(ch)
    ax.set_title('Grad-CAM')    
    gb_viz=gb_vizi     
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(143)
    imgplot = plt.imshow(s_img)
    ax.set_title('GCAM and image')
    print(cam.shape)    
    #gd_gb=cam
    gd_gb = np.dstack((
           gb_vizi[:,:,0]* cam #,
           # gb_viz[:, :, 1] * cam,
           # gb_viz[:, :, 2] * cam,
        ))            
    gb0=np.reshape(gd_gb,[gd_gb.shape[1],gd_gb.shape[1],1])
    gb1=gb0*0.8 +img
    #gd_gb== 255 * gd_gb / np.max(gd_gb)
    ax = fig.add_subplot(144)
    print(gb1.shape,gb0.shape,img.shape)
    #gbw1=np.reshape(gb1,[gd_gb.shape[1],gd_gb.shape[2],gd_gb.shape[0]])
    gb=tf1.keras.preprocessing.image.array_to_img(gb1)
    imgplot = plt.imshow(gb)
    ax.set_title('guided Grad-CAM')

    plt.savefig(store)


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf1.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf1.zeros_like(grad))


#from slim.nets import resnet_v1
#slim = tf.contrib.slim

def layers_print(model,inputs,print_shape_only=False):
     activations=[]
     inp = model.input                              # input placeholder
     outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
     functors = [K.function(inp, [out]) for out in outputs]   # evaluation functions
     # Testin
     list_inputs = inputs

     funcs=functors
     layer_outputs = [func(list_inputs)[0] for func in funcs]
     for layer_activations in layer_outputs:
         activations.append(layer_activations)

     return outputs,functors,activations


def viewer(X,model,store, point=[7,9],model_name='none',height=256,width=256,channels=1,classes=1,case='test',batch=1,label=[0,0,0,1],rot=0,backbone='none'):

    tf1.compat.v1.disable_eager_execution()
    img1 = load_image(X,x=height,y=width,c=channels) #"./demo.png", normalize=False)
    img1=ndimage.rotate(img1,rot,reshape=False)
    batch_size=batch
    print(img1.shape)
    print(img1[:100])
    batch1_img = img1.reshape((1, height, width, channels))
    batch1_label = np.array(label,dtype=np.float32)  # 1-hot result for Boxer
    batch1_label = batch1_label.reshape(1, -1)
    imageK=K.placeholder(shape=(batch_size,height,width,channels))
    label = K.placeholder(shape=(batch_size, classes))
    cn=create_net.create_net(model_name)
    print("the model name is: ",model_name)
    model_structure=cn.net([], [], case,height,channels,(classes),width,backbone) # classes +1 to take into account the background
    print('load weights')
    print("Load model from: ")
    file_store=('%s' %(model))#hdf5
    print(file_store)
    dict1={imageK:batch1_img}
    model_str=model_structure[0]
    model_str.load_weights(file_store)
    #if model_name=="DENRES":
    #     end_p ,functors, image_feature= layers_print(model_str,[batch1_img,batch1_img,batch1_img])
    #else:
    end_p ,functors, image_feature= layers_print(model_str,batch1_img)
    end_points=end_p
      
    print(len(end_points))
    prob =(end_p[(len(end_points)-point[0])])  #,dtype=tf1.float32) # after softmax
    print('prob:', prob)
    cost = (-1) * tf1.reduce_sum(tf1.multiply(batch1_label, tf1.log(prob)), axis=1) #one image per time in other case need modification 
    print('cost:', (cost))
    target_conv_layer =end_p[len(end_points)-point[1]]
    tg=image_feature[len(end_points)-point[1]]
    net=model_str.output
    y_c = tf1.reduce_sum(tf1.multiply(net, batch1_label), axis=1)
    print('y_c:', (y_c))
    print("conv_layer: ", target_conv_layer)
    target_conv_layer_grad = K.gradients(y_c, target_conv_layer)[0]
    # Guided backpropagtion back to input layer
    gb_grad = K.gradients(cost, model_str.input)[0]
    iterate=K.function([model_str.input],[cost,gb_grad])
    iterate2=K.function([target_conv_layer,model_str.input],[y_c,target_conv_layer_grad])
     
    cost_np,gb_grad_value=iterate([batch1_img])
    y_c_np, target_conv_layer_grad_value=iterate2([tg,batch1_img])
    target_conv_layer_value=tg     
    print("result= ",y_c_np)
    y_pred = model_str.predict_generator(batch1_img)
    print("Y_c ",y_c)
    print("Y_C result= ",y_c_np)
    #y_pred = model_str.predict_generator(batch1_img)
    #print("prediction= ",y_pred)
    print("CCN  grad out: ",target_conv_layer_grad_value)
    print("CNN before: ", target_conv_layer)
    print("CNN after: ",target_conv_layer_value)
    print("CNN image: ",tg)




    print("prediction= ",y_pred)    
    for i in range(batch_size):
         visualize(batch1_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gb_grad_value[i],store,x=height,y=width)
    





    
