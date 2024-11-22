# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:37:56 2023

@author: oakley

Make plots of the synthetic maps showing the fold axes.

"""

import geopandas as gp
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

maps_folder = '../Synthetic_Maps/'
results_folder = '../Unsupervised_Clustering_Method/tmp/'
nodes = 20
names = ['FlatModel','FlatModel_EggBoxTopo','SingleFold','SingleFold_EggBoxTopo',
         'AsymmetricFold','SigmoidalFold','MultipleFolds','MultipleFolds_EggBoxTopo']
titles = ['Flat','Flat - EggBox','Single Flold','Single Fold - EggBox',
          'Asymmetric Fold','Sigmoidal Fold','Multiple Folds','Multiple Folds - EggBox']
crs = 'epsg:2154' #Coordinate reference system
strat_num_lims = [2,6] #Minimum and maximum values of the 'StratNum' column that is used to identify the units.
NUM_COLORS = 9#Set this to whatever the maximul number of clusters is.
color_by_fold_type = True #Otherwise, color each axis separately.

#Create the figure.
fig, axs = plt.subplots(2, 4, figsize=(8, 4))
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, hspace=0.05)
cax = plt.axes([0.1, 0.01, 0.8, 0.07])
legend_kwds = {"label": "Relative Age", "orientation": "horizontal", "ticks": [2,3,4,5,6]}
colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, NUM_COLORS)]
if not color_by_fold_type:
    colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, NUM_COLORS)]
else:
    colors = ['blue','red']
newcmp = ListedColormap(colors)

#Loop through the models.
letters = ['A','B','C','D','E','F','G','H']
for i in range(len(names)):
    ind1 = int(i/4)
    ind2 = int(i%4)
    name = names[i]
    
    #Plot the map.
    map_file_path = maps_folder+'/'+name+'/'+name+'.shp'
    geol_map = gp.GeoDataFrame.from_file(map_file_path, geom_col='geometry', crs=crs)
    if i == 0:
        #We only need to add the legend once.
        legend = True
    else:
        legend = False
    colormap = 'Greys'
    geol_map.plot(ax=axs[ind1,ind2], column='RelAge', edgecolor='black', cmap=colormap, 
                  legend=legend, cax=cax, legend_kwds=legend_kwds,
                  vmin=strat_num_lims[0],vmax=strat_num_lims[1])
    if ind2 > 0:
        axs[ind1,ind2].yaxis.set_ticklabels([])
    bbox_props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    axs[ind1,ind2].text(0.1, 0.9, letters[i], horizontalalignment='center',
                        verticalalignment='center', transform=axs[ind1,ind2].transAxes,bbox=bbox_props)
    
    #Plot the fold areas.
    if i >= 2: #For flat models there are no areas, so this will crash looking for the areas .shp file.
        fold_areas_file = results_folder + str(nodes) + '_nodes_' + name + '_fold_area.shp'
        areas_gdf = gp.GeoDataFrame.from_file(fold_areas_file, geom_col='geometry', crs='epsg:2154')
        if not color_by_fold_type:
            areas_gdf.plot(ax=axs[ind1,ind2],categorical=True,cmap=newcmp,legend=False, alpha=0.5, edgecolor='black')
        else:
            areas_gdf.plot(column='fold_type',ax=axs[ind1,ind2],categorical=True,cmap=newcmp,legend=False, alpha=0.5, edgecolor='black')
    
    #Add a scale bar.
    axs[ind1,ind2].set_axis_off() #Don't show the axis numbers.
    axs[ind1,ind2].plot([8e3,9e3],[-1e3,-1e3],'k') #Scale bar
    axs[ind1,ind2].text(9e3,-1e3,'1 km',horizontalalignment='left',verticalalignment='center',fontsize='small')
    
cax.invert_xaxis()
cax.text(0.02, -1.2, 'Older', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)
cax.text(0.98, -1.2, 'Younger', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)

#Save the figure.
fig.savefig('Fig8.pdf',bbox_inches='tight')
fig.savefig('Fig8.png',bbox_inches='tight')