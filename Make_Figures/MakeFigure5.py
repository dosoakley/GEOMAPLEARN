# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:21:10 2023

@author: oakley

Plot the different steps in the process of identifying folds with UNET, using the double fold as an example.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

use_topo = True
img_size = (256,256)
img_size_UNET = (256,256)
trained_model = "../Neural_Network_Method/DetectAreasAndAxisModel"
name = 'DoubleFold'

fig = plt.figure(figsize=(13, 12),dpi=300)
gs = gridspec.GridSpec(3, 3, left=1/13)
axs = []
for i in range(3):
    for j in range(3):
        if not (i==0 and j == 2): #Skip the top left position.
            ax = fig.add_subplot(gs[i,j])
            axs.append(ax)

#Load the trained model.
model = tf.keras.models.load_model(trained_model)

#Read in the data.
if use_topo:
    X = np.zeros((1,img_size_UNET[0],img_size_UNET[1],2))
    n_channels = 2
else:
    X = np.zeros((1,img_size_UNET[0],img_size_UNET[1],1))
    n_channels = 1
y = np.zeros((1,img_size_UNET[0],img_size_UNET[1]))
filename = '../Synthetic_Maps/'+name+'/'+name+'_raster.npz'
npzfile = np.load(filename)
if use_topo:
    X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
    X[0,0:img_size[0],0:img_size[1],1] = npzfile['topo']
else:
    X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
y[0,0:img_size[0],0:img_size[1]] = npzfile['axis'].astype(int) + npzfile['area'].astype(int)
X_orig = X.copy()
                        
#Rescale the data to the range [0,1].
#For the unit numbers, we first shift the numbers so that the non-Quaternary ones start at 1, while Quaternary is 0.
#For topography, we set the minimum topography to 0 and the maximum to 1.
units_unique = np.unique(X[0,:,:,0])
shift = np.min(units_unique[units_unique != 0])-1
X[0,:,:,0][X[0,:,:,0] != 0.0] -= shift
for j in range(1,len(units_unique)):
    shift = units_unique[j] - units_unique[j-1] - 1
    if shift > 0: #There's a gap.
        X[0,:,:,0][X[0,:,:,0] >= units_unique[j]] -= shift
        units_unique[j:] -= shift
X[0,:,:,0] = X[0,:,:,0]/X[0,:,:,0].max()
if use_topo:
    X[0,:,:,1] = (X[0,:,:,1]-X[0,:,:,1].min())/(X[0,:,:,1].max()-X[0,:,:,1].min())
    
#Predicte the areas and axes for the test dataset.
y_pred = model.predict(X)
#This has size (nmodels,img_size[0],img_size[1],2)
#For the last index, channel 1 is chance of a fold axis, 0 is chance of not a fold axis

#Define the text box style for the figure labels.
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
xletter = 10 #In figure coordinates, which are in pixels, from -0.5 to 255.5.
yletter = 232

im1 = axs[0].imshow(X[0,:,:,0].T,origin='lower',cmap='Greys',vmin=0,vmax=1)
axs[0].set_aspect('equal',adjustable='box')
cbar1 = plt.colorbar(im1,ax=axs[0])
cbar1.set_label('Scaled Relative Age',fontsize='large')
axs[0].set_title('Scaled Relative Ages')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].text(xletter,yletter,'A',fontweight='bold',fontsize='large',bbox=props)

im2 = axs[1].imshow(X[0,:,:,1].T,origin='lower',cmap='Greys',vmin=0,vmax=1)
axs[1].set_aspect('equal',adjustable='box')
cbar2 = plt.colorbar(im2,ax=axs[1])
cbar2.set_label('Scaled Elevation',fontsize='large')
axs[1].set_title('Scaled Elevation')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].text(xletter,yletter,'B',fontweight='bold',fontsize='large',bbox=props)

im3 = axs[2].imshow(y_pred[0,:,:,0].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[2].set_aspect('equal',adjustable='box')
cbar3 = plt.colorbar(im3,ax=axs[2])
cbar3.set_label('Probability',fontsize='large')
axs[2].set_title('Class 1 Predicted Probability')
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].text(xletter,yletter,'C',fontweight='bold',fontsize='large',bbox=props)

im4 = axs[3].imshow(y_pred[0,:,:,1].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[3].set_aspect('equal',adjustable='box')
cbar4 = plt.colorbar(im4,ax=axs[3])
axs[3].set_title('Class 2 Predicted Probability')
cbar4.set_label('Probability',fontsize='large')
axs[3].set_xticks([])
axs[3].set_yticks([])
axs[3].text(xletter,yletter,'D',fontweight='bold',fontsize='large',bbox=props)

im5 = axs[4].imshow(y_pred[0,:,:,2].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[4].set_aspect('equal',adjustable='box')
cbar5 = plt.colorbar(im5,ax=axs[4])
axs[4].set_title('Class 3 Predicted Probability')
cbar5.set_label('Probability',fontsize='large')
axs[4].set_xticks([])
axs[4].set_yticks([])
axs[4].text(xletter,yletter,'E',fontweight='bold',fontsize='large',bbox=props)

im6 = axs[5].imshow(y[0,:,:].T==0,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[5].set_aspect('equal',adjustable='box')
cbar6 = plt.colorbar(im6,ax=axs[5])
axs[5].set_title('Class 1 Truth')
cbar6.set_label('Probability',fontsize='large')
axs[5].set_xticks([])
axs[5].set_yticks([])
axs[5].text(xletter,yletter,'F',fontweight='bold',fontsize='large',bbox=props)

im7 = axs[6].imshow(y[0,:,:].T==1,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[6].set_aspect('equal',adjustable='box')
cbar7 = plt.colorbar(im7,ax=axs[6])
axs[6].set_title('Class 2 Truth')
cbar7.set_label('Probability',fontsize='large')
axs[6].set_xticks([])
axs[6].set_yticks([])
axs[6].text(xletter,yletter,'G',fontweight='bold',fontsize='large',bbox=props)

im8 = axs[7].imshow(y[0,:,:].T==2,origin='lower',cmap='viridis',vmin=0,vmax=1)
axs[7].set_aspect('equal',adjustable='box')
cbar8 = plt.colorbar(im8,ax=axs[7])
axs[7].set_title('Class 3 Truth')
cbar8.set_label('Probability',fontsize='large')
axs[7].set_xticks([])
axs[7].set_yticks([])
axs[7].text(xletter,yletter,'H',fontweight='bold',fontsize='large',bbox=props)

#Put labels on the left side for the rows.
axs[0].text(-30,128,'Input to U-Net',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',rotation=90)
axs[2].text(-30,128,'Output from U-Net',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',rotation=90)
axs[5].text(-30,128,'Truth',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',rotation=90)

#Save the figure.
fig.savefig('Fig5.pdf',bbox_inches='tight')
fig.savefig('Fig5.png',bbox_inches='tight')