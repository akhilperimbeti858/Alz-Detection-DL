#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install tf-keras-vis tensorflow


# In[1]:


import numpy as np 
import pandas as pd 
import os
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob
from matplotlib import cm 

import tensorflow as tf 
from tensorflow import keras 
from keras.utils import to_categorical
from keras import layers
from keras import regularizers
from keras import backend as K 
import scipy.misc
from skimage.transform import resize 
from sklearn.metrics import roc_curve, auc


# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[ ]:


tf.__version__, keras.__version__


# ### Getting Image Data

# In[ ]:


patients = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-BINARY/PATIENT_DATA_CI-BINARY.csv')

print("# of patient scans: " + str(len(patients['DEM'])))
scan_paths = patients['PATH']

patients = patients.sample(frac = 1, random_state = 10)
patients = patients.reset_index(drop=True)


# In[ ]:


tr = int(0.7*(len(patients)))
train_data = patients[:tr] # resample to deal with class_imbalance
rem = len(patients) - len(train_data)
val = tr + int(0.2*(len(patients)))
val_data = patients[tr:val]
test_data = patients[val:]
print('Number of paths used in train, val and test arrays are %d, %d, and %d' % (len(train_data), len(val_data), len(test_data)))


# In[ ]:


test_alz1 = test_data[test_data.DEM == 1]
test_alz0 = test_data[test_data.DEM == 0]
print('Test Data Distribution: \n\n Number of patients with Alzheimers: %d \n Number of patients w/out Alzheimers: %d ' % (len(test_alz1), len(test_alz0)))


# In[ ]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)


# ### Pretrained Model - VGG16

# In[ ]:


vgg16_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-BINARY/PRETRAINED_MODELS/DenseNet121/DenseNet121_history_classification_vTEMP')
fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(vgg16_history[metric])
    ax[i].plot(vgg16_history['val_'+ metric])
    ax[i].set_title("VGG16 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[ ]:


vgg16_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-BINARY/PRETRAINED_MODELS/DenseNet121/3D_ALZ_DenseNet121_vTEMP.h5')
vgg16_model.summary()


# ### functions to preprocess images

# In[ ]:


# functions to read and load volume of image data 

def read_3D_MRI(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_3D_MRI(path)
    
    volume = np.stack((volume,)*3, axis=-1) # converting image to RGB
    return volume

# function to display coronal slice of the data
def print_images(path):
    """Displaying axial slice scan"""
    scan = process_scan(path)
    plt.imshow(scan[64,:,:,1])


# ## Saliency - Binary

# In[ ]:


img1 = process_scan(test_paths[3])
img2 = process_scan(test_paths[7])

print(img1.shape)
print_images(test_paths[3])


# In[ ]:


img1 = np.expand_dims(img1,0)
img2 = np.expand_dims(img2,0)


# In[ ]:


print(img1.shape)


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        
        predictions = model(image)
        
        loss = predictions[:, class_idx]
        #loss = categories[np.argmax(predictions)]
        
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    #print(gradient.shape)
    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
    
    return smap

# calling function get_saliency_map

#saliency_map = get_saliency_map(resnet152_model, image = tf.constant(new_img), class_idx = 1)

#channel = 0
#plt.imshow(new_img[0,:,55,:,channel], cmap = 'gray')
#plt.imshow(saliency_map[0,:,55,:,0], alpha = 0.7, cmap = 'seismic')


# In[ ]:


## Saliency for only one image

saliency_map = get_saliency_map(vgg16_model, image = tf.constant(img1), class_idx = 0)

channel = 0
plt.imshow(img1[0,:,55,:,channel], cmap = 'gray')
plt.imshow(saliency_map[0,:,55,:], alpha = 0.4, cmap = 'seismic')


# In[ ]:


resnet152_model.layers[-1].activation = tf.keras.activations.linear # changing activation layer to linear


# In[ ]:


new_img = process_scan(test_paths[25])
new_img.shape


# In[ ]:


new_img = np.expand_dims(new_img,0)
new_img.shape


# In[ ]:


t0 = test_data[test_data.DEM == 0].PATH
t1 = test_data[test_data.DEM == 1].PATH

t0 = t0.reset_index(drop=True)
t1 = t1.reset_index(drop=True)


# #### Class_idx = 0 (Non-Demented)

# In[ ]:


img_array = np.empty((len(t0),128,64))
saliency_arr = np.empty((len(t0),128,64))

for i in range(len(t0)):
    new_img = process_scan(t0[i])
    new_img = np.expand_dims(new_img,0)
    
    
    saliency_map = get_saliency_map(resnet152_model, image = tf.constant(new_img), class_idx = 0)
    
    img_array[i] = new_img[0,:,55,:,0]
    saliency_arr[i] = saliency_map[0,:,55,:]


# In[ ]:


saliency_arr.shape


# In[ ]:


plt.imshow(img_array[3,:,:], cmap = 'gray')
plt.imshow(np.mean(saliency_arr,  axis = 0), alpha = 0.4, cmap = 'seismic') # averaging saliencey maps 


# #### Class_idx = 1 (Demented)

# In[ ]:


img_array = np.empty((len(t1),128,64))
saliency_arr = np.empty((len(t1),128,64))

for i in range(len(t0)):
    new_img = process_scan(t1[i])
    new_img = np.expand_dims(new_img,0)
    
    
    saliency_map = get_saliency_map(resnet152_model, image = tf.constant(new_img), class_idx = 1)
    
    img_array[i] = new_img[0,:,55,:,0]
    saliency_arr[i] = saliency_map[0,:,55,:]


# In[ ]:


plt.imshow(img_array[3,:,:], cmap = 'gray')
plt.imshow(np.mean(saliency_arr,  axis = 0), alpha = 0.3, cmap = 'seismic') # averaging saliencey maps 


# In[ ]:


np.mean(saliency, axis = (0,:,:,:,0)) # averaging saliencey maps 


# In[ ]:





# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        
        loss = predictions[:, class_idx]
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    print(gradient.shape)
    # take maximum across channels
    #gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
    
    return smap

#saliency_map = get_saliency_map(model, image = tf.constant(img_array), class_idx = 0)

#channel = 0
#plt.imshow(img_array[0,:,:,96,channel], cmap = 'gray')
#plt.imshow(saliency_map[0,:,:,96,:], alpha = 0.2)


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# grad-cam method
def make_gradcam_heatmap3D(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# last output activation must be linear
#model.layers[-1].activation = None

# select last convolution layer here
#last_conv_layer_name = [i.name for i in model.layers if 'conv3d' in i.name][-1]
#print(last_conv_layer_name)

# Generate class activation heatmap
#heatmap = make_gradcam_heatmap3D(img_array, model, last_conv_layer_name)

# Display heatmap
#plt.imshow(heatmap[:,:,96])


# In[ ]:


# last output activation must be linear
model.layers[-1].activation = None

# select last convolution layer here
last_conv_layer_name = [i.name for i in model.layers if 'conv3d' in i.name][-1]
print(last_conv_layer_name)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap3D(img_array, model, last_conv_layer_name)

# Display heatmap
plt.imshow(heatmap[:,:,96])

