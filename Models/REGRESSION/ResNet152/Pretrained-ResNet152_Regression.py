#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    
    # converting image to RGB
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


tmp_path = scan_paths[187]
tmp_img = process_scan(tmp_path)
#tmp_img = np.squeeze(tmp_img)
print(tmp_img.shape)

print_images(tmp_path)


# #### MANUAL Train-Val-Test Split (70-20-10)

# In[7]:


## Shuffling rows to reduce bias

patients = patients.sample(frac = 1, random_state = 65)
patients = patients.reset_index(drop=True)
patients


# In[8]:


#from sklearn.model_selection import train_test_split

#train_paths, val_paths = train_test_split(scan_paths,new,test_size=0.20)

tr = int(0.7*(len(patients)))

train_data = patients[:tr] # resample to deal with class_imbalance

rem = len(patients) - len(train_data)

val = tr + int(0.2*(len(patients)))

val_data = patients[tr:val]

test_data = patients[val:]


print('Number of paths used in train, val and test arrays are %d, %d, and %d' % (len(train_data), len(val_data), len(test_data)))


# In[9]:


train_data


# ### Data Generator - CUSTOM

# In[10]:


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


# In[11]:


# Generator instances - Batch size 2

train_generator = data_generator(train_data, 2)
val_generator = data_generator(val_data, 2)


# ### MODEL Implementation - 3D ImageNet Weights

# In[12]:


from classification_models_3D.tfkeras import Classifiers as Classifiers_3D

ResNet152 , preprocess_input = Classifiers_3D.get('resnet152') # 3D ResNet152


# In[13]:


def create_model(input_shape, num_classes):
    
    conv_base = ResNet152(include_top = False,
             weights = 'imagenet',
             input_shape = input_shape,
                     pooling = None)
    
    for layer in conv_base.layers: 
        layer.trainable = False 
            
            
    # Create a new 'top' for the model (its fully connnected layers)
    # Essentially bootstrapping a new top_model onto the pretrained layers
    
    top_model = conv_base.output
    # Flatten matches output dimension of your network with label dimension
    top_model = Flatten()(top_model) # adding flatten 
    
    top_model = Dense(512, activation = 'relu')(top_model)
    top_model = Dense(256, activation = 'relu')(top_model)
    top_model = Dropout(0.3)(top_model)
    
    output_layer = Dense(num_classes, activation = 'linear')(top_model)
    
    
    # Group all layers into a Model() object
    
    model = Model(inputs = conv_base.input, outputs = output_layer, name = '3D_ResNet152')
    
    return model 

# Build model.
resnet152_model = create_model((128,128,64,3), 1)
resnet152_model.summary()


# #### Model Compliation 

# In[14]:


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


# In[15]:


# Compile model.
#initial_learning_rate = 0.0001
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

lr = 0.0001

resnet152_model.compile(
    loss = ccc,
    #loss = 'mse',
    #optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
    optimizer = keras.optimizers.Adam(learning_rate=lr),
    #metrics = keras.metrics.CategoricalAccuracy()
)


# ### Model Fitting

# In[16]:


# Define callbacks

checkpoint_cb = keras.callbacks.ModelCheckpoint("3D_ALZ_ResNet152_v3.h5", save_best_only = True, verbose = 1)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 20, mode = 'min')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                             factor = 0.5, patience = 5, # set to 5
                                             min_lr = 1e-8, min_delta = 1e-7,
                                             verbose = 1, mode = 'min')

csv_log = keras.callbacks.CSVLogger("ResNet152_history_{}_v3".format('regression'), append = True)

# Train the model, doing validation at the end of each epoch
epochs = 200

# start time
start = dt.now()

model_history = resnet152_model.fit(
    train_generator,
    validation_data = val_generator, 
    epochs=epochs,
    steps_per_epoch = len(train_data)/2,
    validation_steps = len(val_data)/2,
    verbose = 1,
    callbacks=[checkpoint_cb, early_stopping_cb, csv_log, reduce_lr]
)

running_secs = (dt.now() - start).seconds


# In[17]:


print('Model took %d seconds to train' % running_secs)


# ## Performance

# In[18]:


resnet152_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/REGRESSION/ResNet152/ResNet152_history_regression_v3')
resnet152_history


# In[ ]:





# In[20]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['loss']):
    ax[i].plot(resnet152_history[metric])
    ax[i].plot(resnet152_history['val_'+ metric])
    ax[i].set_title("ResNet152 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[10]:


resnet152_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-BINARY/PRETRAINED_MODELS/ResNet152/Regression/3D_ALZ_ResNet152_v2.h5',custom_objects = {'ccc': ccc})
resnet152_model.summary()


# In[11]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = resnet152_model.predict(img)
    
    y_pred[i] = pred #categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[12]:


y_pred


# In[13]:


y_true


# In[16]:


y_hat = np.empty(len(test_paths))

for i in range(len(y_pred)):
    if (y_pred[i] >= 0.5):
        y_hat[i] = 1
    else :
        y_hat[i] = 0


# In[17]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_true, y_hat, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['0', '1'],
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    columns = ['No Dementia', "Dementia"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained Resnet152')
plt.xlabel('Predicted CDR')
plt.ylabel('Actual CDR')
plt.show()


# ## Saliency 

# In[20]:


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


# In[18]:


resnet152_model.layers[-1].activation 


# In[19]:


t0 = test_data[test_data.DEM == 0].PATH
t1 = test_data[test_data.DEM == 1].PATH

t0 = t0.reset_index(drop=True)
t1 = t1.reset_index(drop=True)


# In[21]:


img_array = np.empty((len(t0),128,64))
saliency_arr = np.empty((len(t0),128,64))

for i in range(len(t0)):
    new_img = process_scan(t0[i])
    new_img = np.expand_dims(new_img,0)
    
    
    saliency_map = get_saliency_map(resnet152_model, image = tf.constant(new_img), class_idx = 0)
    
    img_array[i] = new_img[0,:,55,:,0]
    saliency_arr[i] = saliency_map[0,:,55,:]
    


# In[22]:


saliency_arr.shape


# In[23]:


plt.imshow(img_array[3,:,:], cmap = 'gray')
plt.imshow(np.mean(saliency_arr,  axis = 0), alpha = 0.4, cmap = 'seismic') # averaging saliencey maps 


# In[ ]:




