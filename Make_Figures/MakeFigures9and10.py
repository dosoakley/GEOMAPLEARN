# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:42:50 2023

@author: oakley

Plot the results on the UNET classification of the maps.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

use_topo = True
img_size = (256,256)
cell_size = 50 #Pixel size in meters.
trained_model = "../Neural_Network_Method/DetectAreasAndAxisModel"

#Names of the model files to test on.
names = ['FlatModel','FlatModel_EggBoxTopo','SingleFold','SingleFold_EggBoxTopo',
         'AsymmetricFold','SigmoidalFold','MultipleFolds','MultipleFolds_EggBoxTopo']

#Load the trained model.
model = tf.keras.models.load_model(trained_model)

#Create the figures.
fig1, axs1 = plt.subplots(2, 4, figsize=(8, 4))
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, hspace=0.05)
cax1 = plt.axes([0.1, 0.01, 0.8, 0.05])
fig2, axs2 = plt.subplots(2, 4, figsize=(8, 4))
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, hspace=0.05)
cax2 = plt.axes([0.1, 0.01, 0.8, 0.05])

#Loop through the models.
letters = ['A','B','C','D','E','F','G','H','I']
for i in range(len(names)):
    #Load the model.
    ind1 = int(i/4)
    ind2 = int(i%4)
    name = names[i]
    if use_topo:
        X = np.zeros((1,img_size[0],img_size[1],2))
        n_channels = 2
    else:
        X = np.zeros((1,img_size[0],img_size[1],1))
        n_channels = 1
    y = np.zeros((1,img_size[0],img_size[1]))
    filename = '../Synthetic_Maps/'+name+'/'+name+'_raster.npz'
    npzfile = np.load(filename)
    if use_topo:
        X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
        X[0,0:img_size[0],0:img_size[1],1] = npzfile['topo']
    else:
        X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
    y[0,0:img_size[0],0:img_size[1]] = npzfile['axis'].astype(int) + npzfile['area'].astype(int)
                        
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
    
    #Plot the true and predicted fold axes and areas side by side.
    ny = X.shape[1] #Number of pixels
    nx = X.shape[2]
    xbar = 0.1*nx #Since we haven't set extent, this is in pixels, not meters.
    ybar = 0.025*ny
    im1 = axs1[ind1,ind2].imshow(y_pred[0,:,:,1].T+y_pred[0,:,:,2].T,origin='lower',vmin=0,vmax=1) #Do this to plot area or axis, since axis is really a part of area.
    axs1[ind1,ind2].plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    axs1[ind1,ind2].text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    axs1[ind1,ind2].set_aspect('equal',adjustable='box')
    axs1[ind1,ind2].xaxis.set_ticks([]) #Ultimately, I should probably have something here, but the pixel number ones are meaningless and overlap the images.
    axs1[ind1,ind2].yaxis.set_ticks([])
    area = npzfile['area'].astype(int)
    axs1[ind1,ind2].contour(area.T,[0.5],colors='black',alpha=0.75) #I could do contour(X,Y,area) without the .T, but then I would need the images to be in the same spatial scale.
    bbox_props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axs1[ind1,ind2].text(0.07, 0.92, letters[i], horizontalalignment='center',
                        verticalalignment='center', transform=axs1[ind1,ind2].transAxes,
                        bbox=bbox_props,fontweight='bold')
    im2 = axs2[ind1,ind2].imshow(y_pred[0,:,:,2].T,origin='lower',vmin=0,vmax=0.4) #Fold axis
    axs2[ind1,ind2].set_aspect('equal',adjustable='box')
    axis = npzfile['axis'].astype(int)
    axs2[ind1,ind2].contour(axis.T,[0.5],colors='black',linewidths=0.5,alpha=0.5)
    axs2[ind1,ind2].xaxis.set_ticks([])
    axs2[ind1,ind2].yaxis.set_ticks([])
    axs2[ind1,ind2].plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    axs2[ind1,ind2].text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    axs2[ind1,ind2].text(0.07, 0.92, letters[i], horizontalalignment='center',
                        verticalalignment='center', transform=axs2[ind1,ind2].transAxes,
                        bbox=bbox_props,fontweight='bold')
    print(y_pred[0,:,:,2].max())
cbar1 = fig1.colorbar(im1, cax=cax1, orientation='horizontal')
cbar1.set_label('Fold Area Probability',fontsize='large')
cbar2 = fig2.colorbar(im2, cax=cax2, orientation='horizontal')
cbar2.set_label('Fold Axis Probability',fontsize='large')

#Save the figures.
fig2.savefig('Fig9.pdf',bbox_inches='tight')
fig2.savefig('Fig9.png',bbox_inches='tight')
fig1.savefig('Fig10.pdf',bbox_inches='tight')
fig1.savefig('Fig10.png',bbox_inches='tight')