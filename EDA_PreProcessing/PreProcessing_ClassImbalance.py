#!/usr/bin/env python
# coding: utf-8

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


data = pd.read_csv('/home/bear/AKHIL_GPU/DATA/oasis_cross-sectional.csv')
patients = data.filter(['ID', 'CDR'], axis = 1)

#new = patients[~patients['CDR'].isna()]
patients['CDR'] = patients['CDR'].fillna(0)

patients = patients.reset_index(drop=True)
patients = patients.drop([119,180],axis = 0) # MRI ID with corrupted images from original dataset
patients = patients.reset_index(drop=True)

new = patients
new


# In[3]:


print(len(patients))


# In[4]:


scan_paths = []

for i in range(len(patients)):
    for f in glob.glob('/home/bear/AKHIL_GPU/Resized_DATA/' + patients.iloc[i,0] + '.nii', recursive = True):
        scan_paths.append(f)
        
print("# of MRI scans: " + str(len(scan_paths)))

#final['PATH'] = scan_paths

scan_paths = np.array(scan_paths)


len(scan_paths)


# In[ ]:


scan_paths = []

for i in range(len(patients)):
        f = glob.glob('/home/bear/AKHIL_GPU/Resized_DATA/' + patients.iloc[i,0] + '.nii', recursive = True)
        if f == []:
            print("File doesnt exist:", '/home/bear/AKHIL_GPU/Resized_DATA/' + patients.iloc[i,0] + '.nii')
        else:
            scan_paths.append(f)            
print("# of MRI scans: " + str(len(scan_paths)))

#final['PATH'] = scan_paths

scan_paths = np.array(scan_paths)


len(scan_paths)
scan_paths


# In[ ]:


scan_paths = []; 

for f in glob.glob('/home/bear/AKHIL_GPU/DATA/**/*anon_sbj_111.img', recursive = True):
    scan_paths.append(f);


print("# of MRI scans: " + str(len(scan_paths)))
scan_paths = np.array(scan_paths)


# ### Train-Val-Test-Split (70-20-10)

# In[ ]:


tr = int(0.7*(len(patients)))

train_data = patients[:tr] # resample to deal with class_imbalance

rem = len(patients) - len(train_data)

val = tr + int(0.2*(len(patients)))

val_data = patients[tr:val]

test_data = patients[val:]


print('Number of paths used in train, val and test arrays are %d, %d, and %d' % (len(train_data), len(val_data), len(test_data)))


# ### Class Imbalance

# #### multi-class

# In[5]:


class0 = new[new['CDR'] == 0]
class1 = new[new['CDR'] == 0.5]
class2 = new[new['CDR'] >= 1]


# In[6]:


class2.shape


# In[7]:


class1_over = class1.sample(n=264, replace = True)
class1_over.shape


# In[8]:


class2_over = class2.sample(n=304, replace = True)
class2_over.shape


# In[9]:


class1_final = class1.append(class1_over, ignore_index=True)

class1_final.shape


# In[10]:


#tmp1 = class2.append(c2a, ignore_index=True)
#tmp2 = tmp1.append(c2b, ignore_index = True)
#tmp3 = tmp2.append(c2c, ignore_index = True)
#class2_final = tmp3.append(class2_over, ignore_index = True)

class2_final = class2.append(class2_over, ignore_index = True)

class2_final.shape


# ##### binary

# In[ ]:


class0 = new[new['CDR'] == 0]
class1 = new[new['CDR'] >= 0.5]
#class2 = new[new['CDR'] >= 1]


# In[ ]:


class1.shape


# In[ ]:


class0.shape


# In[ ]:


class1_over = class1.sample(n=234, replace = True)
class1_over.shape


# In[ ]:


class1_final = class1.append(class1_over, ignore_index=True)

class1_final.shape


# ## Final Data Merging - Multiclass

# In[11]:


fin1 = class0.append(class1_final, ignore_index = True)

final = fin1.append(class2_final, ignore_index = True)

final.shape


# In[12]:


final


# In[13]:


CDR_rating = []

for row in final['CDR']:
    if row == 0.0 :    CDR_rating.append(0)
    elif row == 0.5:   CDR_rating.append(1)
    elif row >= 1.0:  CDR_rating.append(2)
        
    else:           CDR_rating.append('Not_Rated')



final['DEM'] = CDR_rating


scan_paths = []
for i in range(len(final)):
    for f in glob.glob('/home/bear/AKHIL_GPU/Resized_DATA/' + final.iloc[i,0] + '.nii', recursive = True):
        scan_paths.append(f)
print("# of MRI scans: " + str(len(scan_paths)))

final['PATH'] = scan_paths

scan_paths = np.array(scan_paths)


final


# ## Final Data Merging - Binary

# In[ ]:


final = class0.append(class1_final, ignore_index = True)
final.shape


# In[ ]:


final


# In[ ]:


CDR_rating = []

for row in final['CDR']:
    if row == 0.0 :    CDR_rating.append(0)
    elif row >= 0.5:   CDR_rating.append(1)
        
    else:           CDR_rating.append('Not_Rated')



final['DEM'] = CDR_rating


scan_paths = []
for i in range(len(final)):
    for f in glob.glob('/home/bear/AKHIL_GPU/Resized_DATA/' + final.iloc[i,0] + '.nii', recursive = True):
        scan_paths.append(f)
print("# of MRI scans: " + str(len(scan_paths)))

final['PATH'] = scan_paths

scan_paths = np.array(scan_paths)


final


# #### Saving oversampled data to csv

# In[14]:


final.to_csv("PATIENT_DATA_CI_MULTICLASS.csv", index = False) # multiclass Data


# In[ ]:


final.to_csv("PATIENT_DATA_CI_BINARY.csv", index = False) # Binary class data


# #### IMG normalizing and Saving

# In[ ]:


patients = pd.read_csv("PATIENT_DATA_CI.csv")


# In[ ]:


patients = new


# In[ ]:


patients


# In[ ]:


CDR_rating = []

for row in patients['CDR']:
    if row == 0.0 :    CDR_rating.append(0)
    elif row == 0.5:   CDR_rating.append(1)
    elif row >= 1.0:  CDR_rating.append(2)
        
    else:           CDR_rating.append('Not_Rated')



patients['DEM'] = CDR_rating


scan_paths = []
for i in range(len(patients)):
    for f in glob.glob('/home/bear/AKHIL_GPU/DATA/*/' + patients.iloc[i,0] + '/*/*/*/*anon_sbj_111.img', recursive = True):
        scan_paths.append(f)
print("# of MRI scans: " + str(len(scan_paths)))



scan_paths = np.array(scan_paths)


# In[ ]:


img = read_3D_MRI("/home/bear/AKHIL_GPU/DATA/disc4/OAS1_0129_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0129_MR1_mpr_n4_anon_sbj_111.4dfp.img")


# #### FUNCTIONS for RESIZING and NORMALIZING .img files

# In[ ]:


# functions to read and load volume of image data 

def read_3D_MRI(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

# function to normalize image volume 

def normalize(volume):
    """Normalize the volume"""
    volume = np.nan_to_num(volume) 
    min = volume.min()
    max = volume.max()
    volume = (volume - min) / (max - min)
    # normalising to (0-1) and then normalising to 0 mean and 1 std
    volume = (volume - volume.mean())/volume.std()
    volume = volume.astype("float32")
    return volume

# function to resize image volume 

def resize_volume(img):
    #Resize across z-axis
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    current_channel = img.shape[3]

    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor, 1), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_3D_MRI(path)
    #vol_reduced = np.squeeze(vol)
    # Normalize - preprocess_input
    volume = normalize(volume)
    #volume = preprocess_input(vol_reduced)
    # Resize width, height and depth
    volume = resize_volume(volume)
    
    return volume


# In[ ]:


def normalize_and_save(scan_paths, dest_path):
    
    for path in scan_paths:
        
        path_split = path.split('/')
        img_name = path_split[-1][0:13]
        #print(dest_path+img_name+'.nii')
        img = process_scan(path)
        
        for i in range(img.shape[3]):
            axial_image = img[:,:,:,i]
            image = nib.Nifti1Image(axial_image, np.eye(4))
            nib.save(image, dest_path+img_name)


# In[ ]:


dest_path = '/home/bear/AKHIL_GPU/Resized_DATA/'

normalize_and_save(scan_paths, dest_path)


# ### Test if images were saved correctly 

# In[ ]:


new_img = read_3D_MRI('/home/bear/AKHIL_GPU/DATA_FINAL/OAS1_0440_MR1.nii')

print(new_img.shape)

plt.imshow(new_img[:,80,:])


# In[ ]:


new_img = process_scan('/home/bear/AKHIL_GPU/DATA_FINAL/OAS1_0440_MR1.nii')

print(new_img.shape)

plt.imshow(new_img[:,80,:])


# In[ ]:




