#!/usr/bin/env python
# coding: utf-8

# # Pretrained Model Performance - MultiClass Classification

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

import tensorflow as tf 
from tensorflow import keras 
from keras.utils import to_categorical
from keras import layers


# In[2]:


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


# ### Importing data

# In[3]:


patients = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PATIENT_DATA_CI-MULTICLASS.csv')

print("# of patient scans: " + str(len(patients['DEM'])))
scan_paths = patients['PATH']

patients = patients.sample(frac = 1, random_state = 25)
patients = patients.reset_index(drop=True)


# In[4]:


tr = int(0.7*(len(patients)))

train_data = patients[:tr] # resample to deal with class_imbalance

rem = len(patients) - len(train_data)

val = tr + int(0.2*(len(patients)))

val_data = patients[tr:val]

test_data = patients[val:]


print('Number of paths used in train, val and test arrays are %d, %d, and %d' % (len(train_data), len(val_data), len(test_data)))


# In[5]:


train_data


# In[6]:


test_alz2 = test_data[test_data.DEM == 2]
test_alz1 = test_data[test_data.DEM == 1]
test_alz0 = test_data[test_data.DEM == 0]


print('Test Data Distribution: \n\n Number of patients with Moderate/Severe Alzheimers: %d \n Number of patients with Mild Alzheimers: %d \n Number of patients w/out Alzheimers (controls): %d ' % (len(test_alz2),  len(test_alz1), len(test_alz0)))


# ### Pretrained VGG16

# In[7]:


vgg16_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/VGG16/VGG16_history_classification_v2')
vgg16_history


# In[8]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(vgg16_history[metric])
    ax[i].plot(vgg16_history['val_'+ metric])
    ax[i].set_title("VGG16 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])
    
acc = np.asarray(vgg16_history['categorical_accuracy']).mean()
val = np.asarray(vgg16_history['val_categorical_accuracy']).mean()
#print('Mean Training Accuracy: %f' % acc)
#print('Mean Validation Accuracy: %f' % val)


# In[9]:


vgg16_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/VGG16/3D_ALZ_VGG16_v2.h5')
vgg16_model.summary()


# In[12]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = vgg16_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[22]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_pred, y_true, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['No Dementia', "Mild Dementia", "Moderate Dementia"],
                    columns = ['0', '1', '2'])
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True , cmap = 'OrRd')
plt.title('Confusion Matrix - Pretrained VGG16')
plt.ylabel('Predicted CDR')
plt.xlabel('Actual CDR')
plt.show()


# ### Pretrained VGG19

# In[7]:


vgg19_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/VGG19/VGG19_history_classification_v2')
vgg19_history


# In[8]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(vgg19_history[metric])
    ax[i].plot(vgg19_history['val_'+ metric])
    ax[i].set_title("VGG19 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[9]:


vgg19_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/VGG19/3D_ALZ_VGG19_v2.h5')
vgg19_model.summary()


# In[10]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = vgg19_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[17]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_pred, y_true, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['No Dementia', "Mild Dementia", "Moderate Dementia"],
                    columns = ['0', '1', '2'])
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained VGG19')
plt.ylabel('Predicted CDR')
plt.xlabel('Actual CDR')
plt.show()


# ### Pretrained ResNet50

# In[7]:


resnet50_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/ResNet50/ResNet50_history_classification_v2')
resnet50_history


# In[14]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(resnet50_history[metric])
    ax[i].plot(resnet50_history['val_'+ metric])
    ax[i].set_title("ResNet50 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[9]:


resnet50_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/ResNet50/3D_ALZ_ResNet50_v2.h5')
resnet50_model.summary()


# In[10]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = resnet50_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[12]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_pred, y_true, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['No Dementia', "Mild Dementia", "Moderate Dementia"],
                    columns = ['0', '1', '2'])
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained ResNet50')
plt.ylabel('Predicted CDR')
plt.xlabel('Actual CDR')
plt.show()


# ### Pretrained ResNet152

# In[10]:


resnet152_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/ResNet152/ResNet152_history_classification_v2')
resnet152_history


# In[13]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(resnet152_history[metric])
    ax[i].plot(resnet152_history['val_'+ metric])
    ax[i].set_title("ResNet152 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[12]:


resnet152_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/ResNet152/3D_ALZ_ResNet152_v2.h5')
resnet152_model.summary()


# In[14]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = resnet152_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[15]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_true, y_pred, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['0', '1', '2'],
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    columns = ['No Dementia', "Mild Dementia", "Moderate/Severe Dementia"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained Resnet152')
plt.xlabel('Predicted CDR')
plt.ylabel('Actual CDR')
plt.show()


# ### Pretrained DenseNet121

# In[7]:


densenet121_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/DenseNet121/DenseNet121_history_classification_v2')
densenet121_history


# In[8]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(densenet121_history[metric])
    ax[i].plot(densenet121_history['val_'+ metric])
    ax[i].set_title("DenseNet121 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[9]:


densenet121_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/DenseNet121/3D_ALZ_DenseNet121_v2.h5')
densenet121_model.summary()


# In[10]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = densenet121_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[13]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_true, y_pred, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['0', '1', '2'],
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    columns = ['No Dementia', "Dementia", "Moderate Dementia"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained DenseNet121')
plt.xlabel('Predicted CDR')
plt.ylabel('Actual CDR')
plt.show()


# ### Pretrained DenseNet201

# In[7]:


densenet201_history = pd.read_csv('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/DenseNet201/DenseNet201_history_classification_v2')
densenet201_history


# In[8]:


fig, ax = plt.subplots(1,2, figsize=(20,3))
ax = ax.ravel()


for i, metric in enumerate(['categorical_accuracy', 'loss']):
    ax[i].plot(densenet201_history[metric])
    ax[i].plot(densenet201_history['val_'+ metric])
    ax[i].set_title("DenseNet201 Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(['Training', 'Validation'])


# In[9]:


densenet201_model = keras.models.load_model('/home/bear/AKHIL_GPU/FINAL_MODELS/Class_Imbalance-MULTICLASS/PRETRAINED_MODELS/DenseNet201/3D_ALZ_DenseNet201_v2.h5')
densenet201_model.summary()


# In[10]:


test_paths = list(test_data.PATH)
test_dem = list(test_data.DEM)

categories = np.array([0, 1, 2])

y_pred = np.empty(len(test_paths))
y_true = np.empty(len(test_paths))

for i in range(len(test_paths)):
    
    img = process_scan(test_paths[i])
    img = np.expand_dims(img,0)
    
    pred = densenet201_model.predict(img)
    
    y_pred[i] = categories[np.argmax(pred)]
    
    true = test_dem[i]
    
    y_true[i] = true


# In[11]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# normalize must be one of {'true', 'pred', 'all', None}
cm = confusion_matrix(y_true, y_pred, normalize = None)

cm_df = pd.DataFrame(cm,
                    #index = ['0', '0.5', '1', '2'],
                    index = ['0', '1', '2'],
                    #columns = ['Non_Demented', 'Mild_Demented', 'Moderate_Demented', 'Severe_Demented'])
                    columns = ['No Dementia', "Mild Dementia", "Moderate/Severe Dementia"])


plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix - Pretrained DenseNet201')
plt.xlabel('Predicted CDR')
plt.ylabel('Actual CDR')
plt.show()


# In[ ]:




