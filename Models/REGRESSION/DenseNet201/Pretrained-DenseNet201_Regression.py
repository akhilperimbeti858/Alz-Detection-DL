#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import nibabel as nib
from scipy import ndimage
from datetime import datetime as dt

import random
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras 
from keras.utils import to_categorical
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, GlobalAveragePooling3D

import warnings
warnings.filterwarnings('ignore')


# #### Check if GPU backend is being used 

# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# ### Data Preprocessing

# #### FUNCTIONS for READING, RESIZING and NORMALIZING .img files

# In[3]:


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
    
    # converting greyscale image (1 channel) to RGB (3 channel)
    volume = np.stack((volume,)*3, axis=-1) 
    
    
    return volume

# function to display coronal slice of the data
def print_images(path):
    """Displaying coronal slice scan"""
    scan = process_scan(path)
    plt.imshow(scan[64,:,:,1])


# ##### List comprehension - SUBJ_111.img files

# In[4]:


patients = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PATIENT_DATA_CI-MULTICLASS.csv')

print("# of patient scans: " + str(len(patients['DEM'])))
scan_paths = patients['PATH']


# In[5]:


final = patients
CDR_rating = []

for row in final['DEM']:
    if row == 0 :    CDR_rating.append(0.0)
    elif row == 1:   CDR_rating.append(0.5)
    elif row == 2:   CDR_rating.append(1.0)
    #elif row == 2:   CDR_rating.append(1)
        
    else:           CDR_rating.append('Not_Rated')

final['REG'] = CDR_rating
patients = final


# In[6]:


patients


# In[7]:


tmp_path = scan_paths[187]
tmp_img = process_scan(tmp_path)
#tmp_img = np.squeeze(tmp_img)
print(tmp_img.shape)

print_images(tmp_path)


# #### MANUAL Train-Val-Test Split (70-20-10)

# In[8]:


## Shuffling rows to reduce bias

patients = patients.sample(frac = 1, random_state = 65)
patients = patients.reset_index(drop=True)
patients


# In[9]:


#from sklearn.model_selection import train_test_split

#train_paths, val_paths = train_test_split(scan_paths,new,test_size=0.20)

tr = int(0.7*(len(patients)))

train_data = patients[:tr] # resample to deal with class_imbalance

rem = len(patients) - len(train_data)

val = tr + int(0.2*(len(patients)))

val_data = patients[tr:val]

test_data = patients[val:]


print('Number of paths used in train, val and test arrays are %d, %d, and %d' % (len(train_data), len(val_data), len(test_data)))


# In[10]:


train_data


# ### Data Generator - CUSTOM

# In[11]:


def data_generator(df, batch_size, shuffle = True):

    while True:

        if shuffle :
            df = df.sample(frac = 1) # Shuffles paths reducing any inherent bias
            df = df.reset_index(drop = True)
        
        paths = df['PATH']
        dem = df['REG']
        
        batches = int(len(df)/batch_size)

        for index in range(batches):

            batch_paths  = paths[index*batch_size : (index+1)*batch_size] 
            mri_batch_scans = (np.stack([process_scan(filepath) for filepath in batch_paths]))
            batch_CDR = dem[index*batch_size : (index+1)*batch_size]
            batch_DEM = []
            
            for score in batch_CDR:
                
                #val = to_categorical(score, 2)
                batch_DEM.append(float(score)) #make sure to store as float 
                
            yield mri_batch_scans, np.array(batch_DEM) #stops loop until called again


# In[12]:


# Generator instances - Batch size 2

train_generator = data_generator(train_data, 2)
val_generator = data_generator(val_data, 2)


# ### MODEL Implementation - 3D ImageNet Weights

# In[13]:


from classification_models_3D.tfkeras import Classifiers as Classifiers_3D

DenseNet201 , preprocess_input = Classifiers_3D.get('densenet201') # 3D VGG19


# In[14]:


def create_model(input_shape, num_classes):
    
    conv_base = DenseNet201(include_top = False,
             weights = 'imagenet',
             input_shape = input_shape,
                     pooling = 'avg')
    
    for layer in conv_base.layers: 
        layer.trainable = False 
            
            
    # Create a new 'top' for the model (its fully connnected layers)
    # Essentially bootstrapping a new top_model onto the pretrained layers
    
    top_model = conv_base.output
    #top_model = Flatten()(top_model)
    
    top_model = Dense(2048, activation = 'relu')(top_model)
    top_model = Dense(1024, activation = 'relu')(top_model)
    top_model = Dropout(0.3)(top_model)
    
    output_layer = Dense(num_classes, activation = 'linear')(top_model)
    
    
    # Group all layers into a Model() object
    
    model = Model(inputs = conv_base.input, outputs = output_layer, name = '3D_DenseNet201')
    
    return model 

# Build model.
DenseNet201_model = create_model((128,128,64,3), 1)
DenseNet201_model.summary()


# #### Model Compliation 

# In[15]:


def ccc(y_true, y_pred):
    x = y_true; y = y_pred
    
    # mean of truth and predictions
    mx, my = tf.math.reduce_mean(x, axis = 0), tf.math.reduce_mean(y, axis = 0)
    
    # mean centering
    xm,ym = x-mx, y-my
    
    # covariance/variances
    sigxy = tf.math.reduce_mean(tf.multiply(xm,ym), axis = 0)
    sigx, sigy = tf.math.reduce_variance(xm, axis = 0), tf.math.reduce_variance(ym, axis = 0)
    
    # ccc
    num = 2* sigxy
    den = (mx - my)*(mx - my) + sigx + sigy
    
    return -tf.math.reduce_mean(num / den)


# In[16]:


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
lr = 0.0001

DenseNet201_model.compile(
    #loss = 'binary_crossentropy',
    loss = ccc,
    #optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
    optimizer = keras.optimizers.Adam(learning_rate=lr),
    #metrics = keras.metrics.CategoricalAccuracy()
)


# ### Model Fitting

# In[17]:


# Define callbacks

checkpoint_cb = keras.callbacks.ModelCheckpoint("3D_ALZ_DenseNet201_v3.h5", save_best_only = True, verbose = 1)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 20, mode = 'min')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                             factor = 0.5, patience = 5, # set to 5
                                             min_lr = 1e-8, min_delta = 1e-7,
                                             verbose = 1, mode = 'min')

csv_log = keras.callbacks.CSVLogger("DenseNet201_history_{}_v3".format('regression'), append = True)

# Train the model, doing validation at the end of each epoch
epochs = 200

# start time
start = dt.now()

model_history = DenseNet201_model.fit(
    train_generator,
    validation_data = val_generator,
    epochs=epochs,
    steps_per_epoch = len(train_data)/2,
    validation_steps = len(val_data)/2,
    verbose = 1,
    callbacks=[checkpoint_cb, early_stopping_cb, csv_log, reduce_lr]
)

running_secs = (dt.now() - start).seconds


# In[18]:


print('Model took %d seconds to train' % running_secs)


# In[19]:


densenet201_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/REGRESSION/DenseNet201/DenseNet201_history_regression_v3')
densenet201_history #, custom_objects = {'ccc': ccc}


# In[20]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['loss']):
    ax[i].plot(densenet201_history[metric])
    ax[i].plot(densenet201_history['val_'+ metric])
    ax[i].set_title("DenseNet201 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[ ]:




