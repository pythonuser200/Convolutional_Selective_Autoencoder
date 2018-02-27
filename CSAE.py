"""
Codes compiled by Adedotun Akintayo (Graduate Student SCS Lab ME)
copyright 2016: Soumik Sarkar
To use this script, first train the model after all initializations(check the model number in the initialization): 
TRAINING COMMAND: gpu0 python CSAE.py Train_model train_number threshold_val number_of_epochs

VALIDATION COMMAND: gpu0 python CSAE.py Validate_model 2 255 100

And Then make predictions 'NOTE! THE FILE EXTENSION OF THE TEST RESULTS SHOULD BE DIFFERENT FROM TEST DATA: 
TESTING COMMAND: gpu0 python CSAE.py Test_model train_number threshold_val number_of_epochs

* gpu0 - command for specifying what gpu to use in the theano version. 
* train_number is a unique identifier for saving a new training model e.g. 1, 2, 3,...etc. 
* threshold_value is a value between 0 and 255 that helps in the postprocessing for objects and non-object level.
* number_of_epochs is the variable number of runs of the training required e.g. 10, 50, 100, etc.

TO TEST ON NEW DATASETS IN A DIRECTORY, REPLACE THE TRAIN AND TEST PATH IN THE INITIALIZATION TO TEST ON NEW DATASETS
"""
#code compiled by Adedotun Akintayo
#copyright: Soumik Sarkar (SCSLAB ME, 2017)
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>
#from __future__ import division
# add to kfkd.py
import os,skimage,gzip,h5py,lasagne,csv,dill
import cPickle as pickle
from lasagne import layers
from random import sample as samp
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy,theano,pdb,glob
import scipy.ndimage as snd
import theano.tensor as T
from nolearn.lasagne import BatchIterator,TrainSplit
###########################################################################
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne import PrintLayerInfo
##############################################################################
from theano.sandbox.neighbours import neibs2images
import itertools
from shape import ReshapeLayer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import image as im
from skimage.transform import resize
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageEnhance
#from IPython.display import Image as IPImage
from PIL import Image
from pylab import *
import sys,time,string,random
from matplotlib import pyplot as plt
import matplotlib.cm as cm 
import copy
from sys import argv
sys.setrecursionlimit(10000)

##########################################################################
########################### settings for the User here ###################
##########################################################################
patch_size=(16,16)  #Fixed for thie particular model now
te_stride= (2,2) # test stride adjustable in multiples of (2,2), e.g. (4,4), (8,8) up to patch size = (16,16)
result_folder = 'SCNResults/'  #set by the user to save the models
tr_dataname = 'data_patchsize16_16_3to_train.mat'  #name of preprocessed and vectorized training data
te_filename = '%stest_set%s_patch%s_stride%s.hdf5'%(result_folder,'N',patch_size[0],te_stride[0]) #name of preprocessed and vectorized testing data
tr_no = argv[2]   #This is a model identifier number
identifier = "*.jpg"  #Identifier of your training and testing dataset.
save_identifier = "*.png" #NOTE! save_identifier should be differenct from identifier!!!'
#######################################################################################
test_recon_option = ['diffthresandmax'] #other options are ['mean'] or ['diffthresandmean']
thres_val= int(argv[3]) #make this slider adjustable so that some incomplete training may be moved down from [255-180]
#######################################################################################
#path_x ='/home/microway/Lasagne/examples/New_Project/SCNDatasets/SCN images/First 24/S1/S1.02/' running ...160(1)
path_x = 'SCNDatasets/'
comp_tr_va_x1 = True   #This could be made True or false and it is to complement an image
resize_shape1=(480, 640)  #Remind me to change to 0.25 for any other test images. For this ones, it should work fine
groundtruth = False  #if we have the groundtruth for percentage accuracy sake
if groundtruth: #request for path of the groundtruth
   path_y = input('the path to groundtruth is: ')  #'SCNDatasets/testy_resized_2.500000e-01'
comp_tr_va_y1 = True   #The labeled ground truth if it exists. 
#######################################################################################
batch_size=128
conv_filters = 128
deconv_filters = 128
filter_sizes = 3
encode_size =40 #40  ()
epochs = int(argv[4])
##########################################################################
########################### settings for the User here ###################
##########################################################################
class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


class AdjustVariable(object):
    """
    This layer varies the model hyperparameter based on the number of epochs.
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set

class FlipBatchIterator(BatchIterator):
    """
    This layer performs some deformation on the images and their equivalent labels(images)
    """
    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b
        
def remove_all(substr, str):
    index = 0
    length = len(substr)
    while string.find(str, substr) != -1:
        index = string.find(str, substr)
        str = str[0:index] + str[index+length:]
    return str    

def test_patch(patch_size,im_x,im_y,strides,resize_shape,
                complement_x,complement_y,groundtruth):
	xpatches = np.zeros((1,patch_size[0]*patch_size[1]))
	ypatches = np.zeros((1,patch_size[0]*patch_size[1]))
	Ols_images = copy.copy(im_x)
        height, width = np.asarray(Ols_images).shape  
	if groundtruth:
	   Ols_image_y = im_y

	print('test image has height %d and width %d' %(height,width))
	if height < resize_shape[0] or width < resize_shape[1]:# or (height < resize_shape[1] and width < resize_shape[0]):
	  Ols_images = Ols_images.resize((resize_shape[1],resize_shape[0]), Image.BICUBIC) #uses width versus height 

	  if groundtruth is True:
	     Ols_image_y = Ols_image_y.resize((resize_shape[1],resize_shape[0]), Image.BICUBIC)
	     print('test image resized to height %d and width %d' %(np.asarray(Ols_images).shape[1],np.asarray(Ols_images).shape[0]))
	elif height > resize_shape[0] or width > resize_shape[1]:# or (height > resize_shape[1] and width > resize_shape[0]):
	  Ols_images = Ols_images.resize((resize_shape[1],resize_shape[0]), Image.ANTIALIAS)
	     
	  if groundtruth is True:
	     Ols_image_y = Ols_image_y.resize((resize_shape[1],resize_shape[0]), Image.ANTIALIAS)
	     
	#===================================================================================================#

	Ols_images =  np.asarray(Ols_images)
	if groundtruth is True:
	  Ols_image_y = np.asarray(Ols_image_y)
	height, width = Ols_images.shape

	if complement_x == True:
	  Ols_images = imcomplement(Ols_images)
		
	if groundtruth is True and complement_y is True:
	  Ols_image_y = imcomplement(Ols_image_y)    
	     
	#plt.imshow(Ols_images, cmap=cm.binary)
	#plt.figure(1)
	#plt.show()   
	    
	if ((height-patch_size[0]+strides[0])%(strides[0]))== 0:
	  pad_h = 0
	else:
	  pad_h=(strides[0])-((height-patch_size[0]+strides[0])%(strides[0]))
	if ((width-patch_size[1]+strides[1])%(strides[1]))== 0:
	  pad_w = 0
	else:
	  pad_w=(strides[1])-((width-patch_size[1]+strides[1])%(strides[1]))
	endcorr_h = 0 #patch_size[0] 
	endcorr_w = 0 #patch_size[1]
	temp1=np.zeros((height+pad_h+endcorr_h,width+pad_w+endcorr_w))
	if groundtruth is True:
	  temp2=np.zeros((height+pad_h+endcorr_h,width+pad_w+endcorr_w))
	     #print temp1.shape, temp2.shape
	half_corr_h = int(endcorr_h/2)
	half_corr_w = int(endcorr_w/2)
	temp1[half_corr_h:half_corr_h+height,half_corr_w:half_corr_w+width] = Ols_images
	Ols_images = temp1
	if groundtruth is True:
	  temp2[half_corr_h:half_corr_h+height,half_corr_w:half_corr_w+width] = Ols_image_y
	  Ols_image_y = temp2
	     
	height, width = Ols_images.shape
	print(height, width)

	Ols_patche=im.extract_patches(Ols_images, patch_shape=patch_size, extraction_step=strides)
	print(Ols_patche.shape)
	Ols_patches = np.reshape(Ols_patche,(Ols_patche.shape[0]*Ols_patche.shape[1], -1))

	if groundtruth is True:
	  Ols_patche_y = im.extract_patches(Ols_image_y, patch_shape=patch_size, extraction_step=strides)
	  Ols_patches_y = np.reshape(Ols_patche_y,(Ols_patche_y.shape[0]*Ols_patche_y.shape[1], -1)) 
	     ##################################################################
	xpatches = np.concatenate((xpatches,Ols_patches), axis=0)
	if groundtruth is True:
	  ypatches = np.concatenate((ypatches,Ols_patches_y), axis=0)

	xpatches = xpatches[1:xpatches.shape[0],:]
	if groundtruth==True:
	      ypatches = ypatches[1:ypatches.shape[0],:]
	#condition for dataset to take      
	if groundtruth is True:
	  dataset = (xpatches,ypatches)
	elif groundtruth is False:
	  dataset = xpatches
	# Return dataset
	print('... test_set_built')
	dim =np.hstack((height,width,pad_h,pad_w,endcorr_h,endcorr_w))
	return dim,dataset
         
def imcomplement(image):
      """
      Complememt an image
      """
      rval = -image + 255
      return rval
      
def reconstruct_options(patches,te_h, te_w,strides, option,thres_val):
    """
    Function that reconstruct the image from the patches with options
    """
    p_h, p_w = patches.shape[1:3]
    img = np.zeros((te_h,te_w))
    print(img.shape)
    img1 = np.zeros((te_h,te_w))
    i_stride = strides[0]
    j_stride = strides[1]
    # compute the dimensions of the patches array
    n_h = ((te_h - p_h + i_stride)/i_stride).astype(int)
    n_w = ((te_w - p_w + j_stride)/j_stride).astype(int)
    if option=='mean':
       for p, (i, j) in zip(patches, itertools.product(range(n_h), range(n_w))):

           img[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=p
           img1[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=np.ones(p.shape)
       print((img/img1).max(), (img/img1).min())
       result = img/img1
    if option=='diffthresandmean':
       for p, (i, j) in zip(patches, itertools.product(range(n_h), range(n_w))):
           #if (p.max()-p.min()) < thres_val or p.max()<thres_val:
           if p.max()<thres_val:
              p = np.zeros(p.shape)
           img[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=p
           img1[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=np.ones(p.shape)
       result = img/img1

    if option=='diffthresandmax':
       for p, (i, j) in zip(patches, itertools.product(range(n_h), range(n_w))):
            #if (p.max()-p.min()) < thres_val or p.max()<thres_val:
            if p.max()<thres_val:
               p = np.zeros(p.shape)
            img[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=p
            img1 = np.maximum(img1,img)
            img = np.zeros((te_h,te_w))
       print(img1.max(), img1.min())
       result = img1
    return result

def reconstruct_from_patches_with_strides_2d(patches_x,patches_y,image_stat, strides,option,thres_val,groundtruth):
    #te_dim = scipy.io.loadmat(image_stat)
    """
    Function that reconstruct the image from the patches with options
    """
    te_dim = image_stat
    te1_h = te_dim[0].astype(int)
    te1_w = te_dim[1].astype(int)
    pad_h = te_dim[2].astype(int)
    pad_w = te_dim[3].astype(int)
    endcorr_h = te_dim[4].astype(int)
    endcorr_w = te_dim[5].astype(int)

    #i_h, i_w = image_size[:2]
    p_h, p_w = patches_x.shape[1:3]

    #call function for result
    result=reconstruct_options(patches=patches_x,te_h=te1_h, te_w=te1_w,strides=strides, option=option, thres_val = thres_val)

    #remove the endcorrections and pads for both dataset x and y
    half_corr_w = int(endcorr_w/2)
    half_corr_h = int(endcorr_h/2)
    result1 = result[half_corr_h:te1_h-half_corr_h,half_corr_w:te1_w-half_corr_w]

    te_h, te_w = result1.shape

    result = result1[0:(te_h-pad_h),0:(te_w-pad_w)]
    print('original image dimensions are',result.shape)
    #find the error and missed predictions
    re_patche=im.extract_patches(result, patch_shape=(p_h, p_w), extraction_step=(p_h, p_w))
    #print Ols_patche.shape
    re_patches = np.reshape(re_patche,(re_patche.shape[0]*re_patche.shape[1], -1))
    
    if groundtruth is True:
            result_y=reconstruct_options(patches=patches_y,te_h=te1_h, te_w=te1_w,strides=strides, option='mean',thres_val = thres_val)
            result1_y = result_y[half_corr_h:te1_h-half_corr_h,half_corr_w:te1_w-half_corr_w]
            
            # use the new dimension for this parts
            result_y = result1_y[0:(te_h-pad_h),0:(te_w-pad_w)]
            re_patche_y = im.extract_patches(result_y, patch_shape=(p_h, p_w), 
                                             extraction_step=(p_h, p_w))
            re_patches_y = np.reshape(re_patche_y,(re_patche_y.shape[0]*re_patche_y.shape[1], -1))
            
            re_baseline=np.zeros((re_patches.shape[0],re_patches.shape[1]))
	    count1 = 0
	    count0 = 0

	    # label the results roughly
	    label_x=[]

	    for num in xrange(re_patches.shape[0]):
		if (max(re_patches[num,:])>0):
		   label_x.append(1)
		else:
		   label_x.append(0)

	    label_y=[]
	    for num in xrange(re_patches_y.shape[0]):
		if (max(re_patches_y[num,:])>125):
		   label_y.append(1)
		else:
		   label_y.append(0)

	    for num in xrange(len(label_y)):
		if label_x[num] == label_y[num]:
		   count1+=1
		else:
		   count0+=1
	    accuracy = 100.*count1/(count1+count0)
	    print(accuracy)

	    count1 = 0
	    count0 = 0
	    label_base = np.zeros((len(label_y), 1))

	    for num in xrange(len(label_y)):
		if label_base[num] == label_y[num]:
		   count1+=1
		else:
		   count0+=1
	    baselin_accu = 100.*count1/(count1+count0)
	    #print baselin_accu

	    print('accuracy=%0.3f%%_baseline=%0.3f%%'%(accuracy, baselin_accu))  #create some interface for this accuracy
    return result  

#standardize the input dataset
def norm_style1(Y):  
        Y = np.rint(Y * 1).astype(np.int).reshape((-1, 1, patch_size[0], patch_size[1]))  
        Y = Y.astype(np.float64) 
        mu_out, sigma_out = np.mean(Y.flatten()), np.std(Y.flatten())
        Y = (Y -mu_out) / sigma_out 
        Y = Y.astype(np.float32)
        return mu_out, sigma_out, Y 
#normalize the output dataset                       
def norm_style2(Y):
        Y = np.rint(Y * 1).astype(np.int).reshape((-1, 1, patch_size[0], patch_size[1])) 
        X_out = Y.astype(np.float64) 
        mu_out = 127.5
        sigma_out = 127.5
        X_out = (X_out - 125.0) / 125.0
        X_out = X_out.astype(np.float32)
        return mu_out, sigma_out, X_out

                   
def float32(k):
	return np.cast['float32'](k)
    
def get_picture_array(X):
	array = X.reshape(X.shape[0],X.shape[1])
	array = np.clip(array, a_min = 0, a_max = 255)
	return  array  

def load_dataset(patch_size):

        #################################################
        print('... loading training and validation set')##
        #################################################
        f = h5py.File(tr_dataname)
        train_x = np.transpose(f['train_set_x'])
        train_y = np.transpose(f['train_set_y'])
        f.close()
        X = train_x  
        Y = train_y   
        #################################################
	#try the different normalization methods
        mu_out,sigma_out,X_train = norm_style1(X)
       
        _,_, X_out = norm_style2(Y) 
        print(mu_out, sigma_out)
        X_out = X_out.reshape((X_out.shape[0], -1))
        
        print('... maximum value of train_x is ',X_train.max())
        print('... minimum value of train_x is ',X_train.min())
        print('... maximum value of train_y is ',X_out.max())
        print('... minimum value of train_y is ',X_out.min())
        
        return X,Y,X_train,X_out,patch_size, sigma_out,mu_out

# <codecell>
def Train(U_train,v_train):
   ae = NeuralNet(
       layers=[
           ('input', layers.InputLayer),
           ('conv', layers.Conv2DLayer),
           ('conv1', layers.Conv2DLayer),
           ('pool', layers.MaxPool2DLayer),
           ('dropout1', layers.DropoutLayer),
           ('conv2', layers.Conv2DLayer),
           ('pool1', layers.MaxPool2DLayer),
           ('dropout2', layers.DropoutLayer),
           ('flatten', ReshapeLayer),  # output_dense
           ('encode_layer', layers.DenseLayer),
           ('dropout3', layers.DropoutLayer),
           ('hidden', layers.DenseLayer),  
           ('unflatten', ReshapeLayer),
           ('unpool', Unpool2DLayer),
           ('deconv', layers.Conv2DLayer),
           ('output_layer', ReshapeLayer),
           ],
       input_shape=(None, 1, patch_size[0], patch_size[1]),
       conv_num_filters=conv_filters, conv_filter_size = (filter_sizes, filter_sizes),
       conv1_num_filters=conv_filters, conv1_filter_size = (filter_sizes, filter_sizes),
       conv2_num_filters=conv_filters, conv2_filter_size = (filter_sizes, filter_sizes),
       conv_nonlinearity=None,
       pool_pool_size=(2, 2),
       pool1_pool_size = (2, 2),
       dropout1_p=0.3,
       dropout2_p=0.3,
       dropout3_p=0.2,
       flatten_shape=(([0],-1)),
       encode_layer_num_units = encode_size,
       hidden_num_units= deconv_filters * (patch_size[0] + filter_sizes - 1) ** 2 / 4,
       train_split=TrainSplit(eval_size=0.2),
       unflatten_shape=(([0], deconv_filters, (patch_size[0] + filter_sizes - 1) / 2, (patch_size[1] + filter_sizes - 1) / 2 )),
       unpool_ds=(2, 2),
       deconv_num_filters=1, deconv_filter_size = (filter_sizes, filter_sizes),
       deconv_nonlinearity=None,
       output_layer_shape = (([0],-1)),
       update_learning_rate = 0.002,
       update_momentum = 0.975,
       batch_iterator_train=FlipBatchIterator(batch_size=batch_size),  
       regression=True,
       max_epochs= epochs,
       verbose=2,
       )
   ae.fit(U_train, v_train)
   return ae 

#Newly added by Dotun for training the model here ###############################################
def Train_model(patch_size=patch_size):

	X,Y,X_train,X_out,patch_size,sigma,mu = load_dataset(patch_size=patch_size)
	ae = Train(U_train=X_train,v_train=X_out)
    
	pickle.dump(ae, open('%s/pa_si%str%s_ae.pkl'%(result_folder, patch_size[0],tr_no),"wb"))

	X_train_pred = ae.predict(X_train).reshape(-1, patch_size[0], patch_size[1]) * 127.5 + 127.5

	X_pred = np.rint(X_train_pred).astype(int) #seems like does not matter
	X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
	
	X_pred = X_pred.astype('uint8')


def Validate_model(patch_size=patch_size):
	print('loading the model')
	#tr_no = '25_2'      
	###########################################################################################
	################  Plot loss for two functions here ########################################
	###########################################################################################

	f = open('%s/pa_si%str%s_ae.pkl'%(result_folder, patch_size[0],tr_no),'rb')
	ae = pickle.load(f)
	f.close()

	fig = plt.figure()
	train_loss2 = np.array([i["train_loss"] for i in ae.train_history_])
	valid_loss2 = np.array([i["valid_loss"] for i in ae.train_history_])   



	plt.plot(train_loss2,"b--", linewidth=3,label="M2 training loss")
	plt.plot(valid_loss2,"b", linewidth=2,label="M2 validation loss")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch",fontsize=16)
	plt.ylabel("loss",fontsize=16)
	plt.legend(['train', 'validation'], loc='upper left')
	fig.savefig('%s/loss_pa_%s_tr_%s.eps'%(result_folder,  patch_size[0],tr_no))

	#plot_loss(ae)
	#plt.savefig('%s/losstr%ste%s' %(result_folder,tr_no,test_index), ext="png", close=True)
	#plt.close()   

	print('loss plot saved')
		#IPImage('%s/test%s.png'%(result_folder,test_index)) 

	print('loading patched data')
	X,Y,X_train,X_out,patch_size,sigma,mu = load_dataset(patch_size=patch_size)
	print(Y.max())

	train_index = samp(range(0, X.shape[0]), 1)[0] #choose 32 random filters

	x = X_train[train_index:train_index+1]
	print(x.shape)
	X_train_pred = ae.predict(X_train).reshape(-1, patch_size[0], patch_size[1])* 127.5 + 127.5

	X_pred = np.rint(X_train_pred).astype(int) #seems like does not matter
	X_pred = np.clip(X_pred, a_min=0, a_max=255)
	X_pred = X_pred.astype('uint8')

	#get_random_images1(X=X,Y=Y,X_pred=X_pred,filename='%s/example_%s_pa_%s_tr_%s.eps'%(result_folder,train_index, patch_size[0],tr_no), index=index_set,patch_size=patch_size)
	#conv
	plot_conv_weights(ae.layers_['conv'], figsize=(4,4))
	plt.savefig('%s/convW_pa_%s_tr_%s'%(result_folder, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	plot_conv_activity(ae.layers_['conv'], x)
	plt.savefig('%s/convAct_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')
	#conv1
	plot_conv_weights(ae.layers_['conv1'],figsize=(4,4))
	plt.savefig('%s/conv1W_pa_%s_tr_%s'%(result_folder, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	plot_conv_activity(ae.layers_['conv1'],x)
	plt.savefig('%s/conv1Act_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close()
	#pool
	plot_conv_activity(ae.layers_['pool'],x)
	plt.savefig('%s/poolAct_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	#conv2
	plot_conv_weights(ae.layers_['conv2'],figsize=(4,4))
	plt.savefig('%s/conv2W_pa_%s_tr_%s'%(result_folder, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	plot_conv_activity(ae.layers_['conv2'],x)
	plt.savefig('%s/conv2Act_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	#pool1
	plot_conv_activity(ae.layers_['pool1'],x)
	plt.savefig('%s/pool1Act_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	#Full connection layer
	W_code = ae.layers_['encode_layer'].W.get_value(borrow=True)
	scipy.misc.imsave('%s/encode_weight_tr%ste%s.png'%(result_folder,tr_no,train_index), W_code[0:1000,:])

	W_hidden = ae.layers_['hidden'].W.get_value(borrow=True)
	scipy.misc.imsave('%s/hidden_weight_tr%ste%s.png'%(result_folder,tr_no,train_index), W_code[0:1000,:])

	#unpool
	plot_conv_activity(ae.layers_['unpool'], x)
	plt.savefig('%s/unpoolAct_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	#unpool
	plot_conv_activity(ae.layers_['deconv'], x)
	plt.savefig('%s/deconvAct_%s_pa_%s_tr_%s'%(result_folder,train_index, patch_size[0],tr_no), ext="png", close=True)
	plt.close('all')

	#deconv
	#plot_conv_weights(ae.layers_['deconv'],figsize=(4,4))
	#plt.savefig('%s/deconvW_pa_%s_tr_%s'%(result_folder, patch_size[0],tr_no), ext="png", close=True)
	#plt.close('all')
	#deconv
	xs = T.tensor4('xs').astype(theano.config.floatX)
	get_activity = theano.function([xs], get_output(ae.layers_['deconv'], xs))
	deconv_activity = np.reshape(get_activity(x),(patch_size[0], patch_size[1]))
	scipy.misc.imsave('%s/deconv_act_tr%ste%s.png'%(result_folder,tr_no,train_index),deconv_activity)
####################################################################################   
################################## <Inference codecell> ############################
####################################################################################
def  test_data(patch_size,im_x,im_y,te_stride,resize_shape,complement_x,complement_y,groundtruth):
        
	dim,dataset = test_patch(patch_size=patch_size,im_x=im_x,im_y=im_y,strides=te_stride,resize_shape=resize_shape,
		        complement_x=complement_x,complement_y=complement_y,groundtruth=groundtruth)

        if groundtruth is True:
		U, v = dataset
		_,_, U_train = norm_style1(U)
		#U_train = normx(U) 
		mu_t, sigma_t, v_train = norm_style1(v) 
		v_train = v_train.reshape((v_train.shape[0], -1))
        else:   
                sigma_t = 127.5  # JUST TAKE THIS TO BE FOR THE TEST SET
                mu_t = 127.5  # JUST TAKE THIS TO BE FOR THE TEST SET
                U = dataset
                _, _, U_train = norm_style1(U)
                #U_train = spec_normx(U)  
                v_train = np.zeros((U.shape))   #dummy v_train needed to load training data
                v = np.zeros((v_train.shape))      # create another dummy v      
                v_train = v_train.reshape((v_train.shape[0], -1))        
        return U_train,v_train,U,v,dim,mu_t,sigma_t
      
      
###########################################################################
#codecell starts here for all the images in a certain folder
def Test_model():
        print('NOTE! USE DIFFERENT EXTENSIONS FOR THE TEST RESULTS FROM THE TEST DATA OR THEY WILL BE OVERWRITTEN')
	N_eggs = []  # gettting the dataframe for number of eggs
	name = []  #get the names of the data

	for infile in glob.glob(os.path.join(path_x, identifier)):
		Ols_images = Image.open (infile).convert('L')
		if groundtruth is True:
		     A = remove_all(path_x,infile)
		     img_y = path_y + A
		     print(img_y)
		     Ols_image_y = Image.open (img_y).convert('L')
		else:
		     Ols_image_y = np.zeros((np.asarray(Ols_images).shape))
			     
		t0 =time.time()
		# call the test_data function here

		U_train, v_train,U, v,dim,mu_t, sigma_t = test_data(patch_size=patch_size,im_x=Ols_images,im_y=Ols_image_y,
				    te_stride=te_stride,resize_shape=resize_shape1,complement_x=comp_tr_va_x1,
				    complement_y=comp_tr_va_y1,groundtruth=groundtruth)
	
		if 'ae' in globals():
		        ae = ae
		else:
				#if known_model is None:	    
				f = open('%s/pa_si%str%s_ae.pkl'%(result_folder, patch_size[0],tr_no),'rb')
				ae = pickle.load(f)
				f.close()			
				"""else:
				f = open(known_model,'rb')
				ae = pickle.load(f)
				f.close()"""	
		print('... loading egg data testing set')
		print('... maximum value of testing is ',U_train.max())
		print('... minimum value of testing is ',U_train.min())

		t1 =time.time()
		U_pred = ae.predict(U_train).reshape((-1, 1, patch_size[0], patch_size[1])) 
		U_train_pred = U_pred.reshape(-1, patch_size[0], patch_size[1])*sigma_t+mu_t
		U_pred = np.rint(U_train_pred).astype(int)
		U_pred = np.clip(U_pred, a_min = 0, a_max = 255)

		print('... maximum value of test prediction is ',U_pred.max())
		print('... minimum value of test prediction is ',U_pred.min())
		U_pred = U_pred.astype('uint8')
		if groundtruth is True:
			 patches_y=np.reshape(v,(U_pred.shape[0],U_pred.shape[1], U_pred.shape[2]))
		else:
			 patches_y=np.zeros((U_pred.shape[0], U_pred.shape[1], U_pred.shape[2]))


		######################################################################################
		##### save the predictions for the slider of thresholding here########################
	
	
		for opts in xrange(len(test_recon_option)):
			img_name2 = '%smodel_no%s_thres_val%dpatch=%d_stride=%d'%(test_recon_option[opts],tr_no,thres_val,patch_size[0],te_stride[0])
			rec_c = reconstruct_from_patches_with_strides_2d(patches_x=U_pred,patches_y=patches_y,
				           image_stat=dim,strides=te_stride, option=test_recon_option[opts],thres_val=thres_val,
				           groundtruth=groundtruth)
			#copy the reconstructed results
		        ans = copy.copy(rec_c)
				#---------------------------SCN Counting --------------------------------------#
				# Counting the number of objects using a connected components. This may be commented if not needed
		        ans[ans <= 50] = 0  
		        n_of_eggs = snd.label(ans)[1] 
				#---------------------------Comment above lines if not counting--------------------------#
		        t2 =time.time()
		        print (" Algorithm on %d patches plus patching took %f seconds" %(U_pred.shape[0],t2-t0))
			print (" Algorithm on %d patches took %f seconds" %(U_pred.shape[0],t2-t1))
		        ########### get the image saved into file here ####################################
		        original_image = Image.fromarray(get_picture_array(rec_c))
		        new_size = (original_image.size[0], original_image.size[1])
		        new_im = Image.new('L', new_size)
		        new_im.paste(original_image, (0,0))
		        new_im.save(os.path.splitext(infile)[0]+img_name2+save_identifier, format="%s"(save_identifier.replace(".", "")))


		N_eggs.append(n_of_eggs)  #dataframe for number of eggs
		fileName = os.path.split(infile)[1]
		name.append(fileName)

	#---------------------------COMMENT OUT IF NOT COUNTING OBJECTS ------------------#
	name.append('total count')
	N_eggs.append(np.sum(N_eggs)) #append the sum of eggs

	df = DataFrame({"Img_ID": Series(name), "Machine_count":Series(N_eggs)})
	df.to_csv(path_x+"/Result_tr%str%sthresh%s.csv"%(tr_no, te_stride[0],thres_val), index=False) 
	#---------------------------COMMENT ABOVE LINES IF NOT COUNTING-------------------#                   
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func() #(*sys.argv[2:])
                   
