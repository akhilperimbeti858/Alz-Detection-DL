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
from tensorflow.keras import layers 



# ## Classification

# In[2]:


patients = pd.read_csv('/home/bear/AKHIL_GPU/DATA/oasis_cross-sectional.csv')


#new = patients[~patients['CDR'].isna()]
patients['CDR'] = patients['CDR'].fillna(0)

patients = patients.reset_index(drop=True)
patients = patients.drop([119,180],axis = 0) # MRI ID with corrupted images from original dataset
patients = patients.reset_index(drop=True)

final = patients.filter(['ID', 'CDR'], axis = 1)

CDR_rating = []

for row in final['CDR']:
    if row == 0.0 :    CDR_rating.append(0)
    elif row == 0.5:   CDR_rating.append(1)
    elif row >= 1.0:  CDR_rating.append(2)
        
    else:           CDR_rating.append('Not_Rated')



final['DEM'] = CDR_rating


# In[10]:


print('Distribution of the Classes in the subsample dataset')
print(final['DEM'].value_counts()/len(final))

colors = ["#0101DF", "#DF0101", "#8A2BE2"]

plt.figure(figsize=(10,5))

ax = sns.countplot(data=final, x = 'DEM', hue = 'DEM', palette=colors)

for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.32, p.get_height()+0.9))

plt.legend(labels = ["Class 0", "Class 1", "Class 2"])
plt.title("Class Distributions \n (0: Control || 1: Mild AD || 2: Moderate/Severe AD)", fontsize=10)



# ### Correlation between age and Normalized whole brain volume

# In[4]:


age_dem0 = patients.Age[patients.CDR == 0.0]
age_dem1 = patients.Age[patients.CDR == 0.5]
age_dem2 = patients.Age[patients.CDR >= 1.0]

nWBV_dem0= patients.nWBV[patients.CDR == 0.0]
nWBV_dem1= patients.nWBV[patients.CDR == 0.5]
nWBV_dem2 = patients.nWBV[patients.CDR >= 1.0]

plt.scatter(age_dem0, nWBV_dem0, c ='green')
plt.scatter(age_dem1, nWBV_dem1, c ='yellow')
plt.scatter(age_dem2, nWBV_dem2, c ='red')
plt.xlabel("Patient Age")
plt.ylabel("Normalized Whole Brain Volume")
plt.title('Distribution of Demented and Non-Demted by Age vs. nWBV', fontsize = 12)
plt.legend(['Control (No AD): CDR = 0', 'MCI/Mild AD: CDR = 0.5', 'Moderate/Severe AD: CDR >= 1.0'], fontsize = 8)
plt.show()


# ## Image EDA

# In[5]:


IDS = ['OAS1_0005_MR1', 'OAS1_0453_MR1', 'OAS1_0052_MR1', 'OAS1_0351_MR1'] ## CDR = 0, 0.5, 1, 2


# ### Full images - (256 x 256 x 128 x 1)

# In[6]:


tmp_paths_full = []; 

for i in range(len(IDS)):
    for f in glob.glob('/home/bear/AKHIL_GPU/DATA/*/' + IDS[i] + '/**/*anon_sbj_111.img', recursive = True):
        tmp_paths_full.append(f);


print("# of MRI scans: " + str(len(tmp_paths_full)))
tmp_paths_full = np.array(tmp_paths_full)


# In[7]:


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
    
  
    return volume

# function to display axial, coronal or sagittal  slice of the data
def print_images1(path, num, plane ):
    """Displaying axial slice scan"""
    scan = process_scan(path)
    
    if plane == 'axial':
        plt.imshow(scan[:,num,:,0])
    elif plane == 'coronal':
        scan = np.rot90(scan, 2)
        plt.imshow(scan[num,:,:,0])
    elif plane == 'sagittal':
        scan = np.rot90(scan, 1)
        plt.imshow(scan[:,:,num,0])
    else:
        print('Error: Please specify proper plane.')
        
# function to display axial, coronal or sagittal slice of the data
def print_images2(path, num, plane ):
    """Displaying axial slice scan"""
    scan = process_scan(path)
    
    if plane == 'axial':
        plt.imshow(scan[:,num,:], cmap = 'gray')
    elif plane == 'coronal':
        plt.imshow(scan[num,:,:])
    elif plane == 'sagittal':
        plt.imshow(scan[:,:,num])
    else:
        print('Error: Please specify proper plane.')
    
    
    


# #### Patient ID: 0351_MR1 - Diagnosed with CDR = 2, Moderate/Severe AD

# In[10]:


print_images1(tmp_paths_full[3], 165, 'axial')


# In[24]:


print_images1(tmp_paths_full[3], 128, 'coronal')


# In[25]:


print_images1(tmp_paths_full[3], 64, 'sagittal')


# #### Patient ID: 0005_MR1 - Diagnosed with CDR = 0, Control, No AD

# In[17]:


print_images1(tmp_paths_full[0], 130, 'axial')


# In[36]:


print_images1(tmp_paths_full[0], 120, 'coronal')


# In[35]:


print_images1(tmp_paths_full[0], 64, 'sagittal')


# #### Patient_ID: OAS1_0453_MR1: Diagnosed CDR = 0.5, MCI, Very Mild AD

# In[32]:


print_images1(tmp_paths_full[1], 128, 'axial')


# In[33]:


print_images1(tmp_paths_full[1], 125, 'coronal')


# In[34]:


print_images1(tmp_paths_full[1], 70, 'sagittal')


# #### Patient_ID: OAS1_0052_MR1: Diagnosed CDR = 1,  Mild/Moderate AD

# In[35]:


print_images1(tmp_paths_full[2], 155, 'axial')


# In[36]:


print_images1(tmp_paths_full[2], 128, 'coronal')


# In[37]:


print_images1(tmp_paths_full[2], 70, 'sagittal')


# ### Normalized and Resized Images - (128 x 128 x 64 x 1)  

# In[139]:


tmp_paths_resized = []; 

for i in range(len(IDS)):
    for f in glob.glob('/home/bear/AKHIL_GPU/Resized_DATA/' + IDS[i] + '.nii'):
        tmp_paths_resized.append(f);


print("# of MRI scans: " + str(len(tmp_paths_resized)))
tmp_paths_resized = np.array(tmp_paths_resized)


# In[152]:


print_images2(tmp_paths_resized[0], 64, 'coronal')


# In[183]:


print_images2(tmp_paths_resized[0], 60, 'axial')


# In[199]:


print_images2(tmp_paths_resized[0], 26, 'sagittal')


# #### Montage of Image Slices - (3D MRI Scan) - Sagittal Plane view

# In[8]:


def plot_slices(num_rows, num_cols, width, height, data):
    "Plotting a montage of 2D slices"
    data = np.rot90(np.array(data),2)
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_cols, width, height))
    rows_data, cols_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 15
    fig_height = fig_width + sum(heights) / sum(widths)
    
    f,axrr = plt.subplots(rows_data,cols_data,
                         figsize = (fig_width,fig_height),
                          gridspec_kw = {"height_ratios":heights})
    
    for i in range(rows_data):
        for j in range(cols_data):
            #axrr[i,j].imshow(data[i][j], cmap="cool")
            axrr[i,j].imshow(data[i][j])
            axrr[i,j].axis("off")
            
    plt.subplots_adjust(wspace = 0, hspace = 0, left = 0, right = 1, bottom = 0, top = 1)
    plt.show()


# In[9]:


img_control = process_scan(tmp_paths_full[0])
img_control.shape


# In[10]:


#64 slices 

plot_slices(8,8, 256, 256, img_control[:,:,32:96,0] )


# In[11]:


img_dem = process_scan(tmp_paths_full[3])
img_dem.shape


# In[12]:


#64 slices 

plot_slices(8,8, 256, 256, img_dem[:,:,32:96,0] )


# In[ ]:




