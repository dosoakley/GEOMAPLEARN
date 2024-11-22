# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:37:56 2023

@author: oakley

Make plots of the synthetic maps.
"""

import geopandas as gp
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

maps_folder = '../Synthetic_Maps/'
names = ['FlatModel','FlatModel_EggBoxTopo','SingleFold','SingleFold_EggBoxTopo',
         'AsymmetricFold','SigmoidalFold','MultipleFolds','MultipleFolds_EggBoxTopo']
titles = ['Flat','Flat - EggBox','Single Flold','Single Fold - EggBox',
          'Asymmetric Fold','Sigmoidal Fold','Multiple Folds','Multiple Folds - EggBox']
crs = 'epsg:2154' #Coordinate reference system
strat_num_lims = [2,6] #Minimum and maximum values of the 'StratNum' column that is used to identify the units.

#Create the figure.
fig = plt.figure(figsize=(10, 4),dpi=300)
gs1 = gridspec.GridSpec(2,5,bottom=0.1)
gs1.update(hspace=0.02)
axs = []
for i  in range(2):
    for j in range(4):
        ax = fig.add_subplot(gs1[i,j])
        axs.append(ax)
cax = plt.axes([0.14, 0.01, 0.6, 0.05])
ax2_1 = fig.add_subplot(gs1[0,4])
ax2_2 = fig.add_subplot(gs1[1,4])
axs2 = [ax2_1,ax2_2]
cax2 = plt.axes([0.77, 0.01, 0.13, 0.05])

legend_kwds = {"label": "Relative Age", "orientation": "horizontal", "ticks": [2,3,4,5,6]}

#Loop through the models.
letters = ['A','B','C','D','E','F','G','H']
for i in range(len(names)):
    name = names[i]
    file_path = maps_folder+'/'+name+'/'+name+'.shp'
    geol_map = gp.GeoDataFrame.from_file(file_path, geom_col='geometry', crs=crs)
    #Plot the map.
    if i == 0:
        #We only need to add the legend once.
        legend = True
    else:
        legend = False
    geol_map.plot(ax=axs[i], column='RelAge', edgecolor='black', 
                  legend=legend, cax=cax, legend_kwds=legend_kwds,
                  vmin=strat_num_lims[0],vmax=strat_num_lims[1])
    # Make the figure look nice.
    if i > 4:
        axs[i].yaxis.set_ticklabels([])
    bbox_props = dict(boxstyle='square', facecolor='white', alpha=0.7)
    axs[i].text(0.1, 0.9, letters[i], horizontalalignment='center',
                        verticalalignment='center', transform=axs[i].transAxes,bbox=bbox_props)
    axs[i].set_axis_off() #Don't show the axis numbers.
    axs[i].plot([8e3,9e3],[-1e3,-1e3],'k') #Scale bar
    axs[i].text(9e3,-1e3,'1 km',horizontalalignment='left',verticalalignment='center',fontsize='small')
cax.invert_xaxis()
cax.text(0.02, -1.2, 'Older', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)
cax.text(0.96, -1.2, 'Younger', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)

#Plot the elevations as well.
letters2 = ['I','J']
# step_sizes = [100,300] #For contours
for i in range(2): #The first two maps give the two different topographies, so we'll use those.
    name = names[i]
    npz_file_path = maps_folder+'/'+name+'/'+name+'_topo.npz'
    npz_file = np.load(npz_file_path)
    X = npz_file['X']
    Y = npz_file['Y']
    topo = npz_file['topo']
    [xmin,xmax,ymin,ymax] = [X.min()-10,X.max()+10,Y.min()-10,Y.max()+10] #Assuming that the topo values are for the center of each pixel, and each pixel is 20 x 20 m.
    im = axs2[i].imshow(topo.T,extent=[xmin,xmax,ymin,ymax],cmap='cividis',vmin=2900,vmax=3900,origin='lower')
    cbar = plt.colorbar(im,cax=cax2,orientation='horizontal',ticks=[2900,3900])
    cbar.set_label('Elevation (m)')
    xlim, ylim = [axs[i].get_xlim(),axs[i].get_ylim()]
    axs2[i].set_xlim(xlim[0],xlim[1])
    axs2[i].set_ylim(ylim[0],ylim[1])
    axs2[i].set_axis_off()
    border = plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False, color='k')
    axs2[i].add_patch(border)
    axs2[i].text(0.06, 0.9, letters2[i], horizontalalignment='center',
                        verticalalignment='center', transform=axs2[i].transAxes,bbox=bbox_props)
    axs2[i].plot([8e3,9e3],[-1e3,-1e3],'k') #Scale bar
    axs2[i].text(9e3,-1e3,'1 km',horizontalalignment='left',verticalalignment='center',fontsize='small')


#Add boxes around the two parts of the figure.
fig.patches.extend([plt.Rectangle((0.12,-0.1),0.63,1.0,
                                  fill=False, color='k',transform=fig.transFigure, figure=fig)])
fig.patches.extend([plt.Rectangle((0.75,-0.1),0.17,1.0,
                                  fill=False, color='k',transform=fig.transFigure, figure=fig)])

#Save the figure.
fig.savefig('Fig6.pdf',bbox_inches='tight')
fig.savefig('Fig6.png',bbox_inches='tight')