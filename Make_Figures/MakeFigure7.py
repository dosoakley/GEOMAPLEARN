# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:37:56 2023

@author: oakley

Make plots of the synthetic maps showing the fold axes.

Instead of a separate plot for each axis, I might want one color for anticlines and one color for synclines.
Possibly even a separate set of plots for anticlines and synclines, at least for the fold areas.
However, at present axes_gdf doesn't include anticline vs. syncline information, so I would need to save that to the files.

"""

import geopandas as gp
from matplotlib import pyplot as plt
import pandas as pd
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
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, hspace=0.02)
cax = plt.axes([0.1, 0.01, 0.8, 0.05])
legend_kwds = {"label": "Relative Age", "orientation": "horizontal", "ticks": [2,3,4,5,6]}
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
    
    #Plot the fold axes.
    if i >= 2: #For flat models there are no axes, so this will crash looking for the axes .shp file.
        fold_axes_file = results_folder + str(nodes) + '_nodes_' + name + '_fold_axes.shp'
        axes_gdf = gp.GeoDataFrame.from_file(fold_axes_file, geom_col='geometry', crs='epsg:2154')
        clusters_unique = pd.unique(axes_gdf['cluster']).tolist()
        for j,c in enumerate(clusters_unique):
            ind = axes_gdf.index[axes_gdf.cluster==c][0] #This should only return one item.
            x,y = axes_gdf.geometry[ind].xy
            # axs[ind1,ind2].plot(x,y,'-')
            if color_by_fold_type:
                color = colors[axes_gdf.fold_type[ind]]
            else:
                color = colors[j]
            axs[ind1,ind2].plot(x,y,'-',color=color)
    
    #Add a scale bar.
    axs[ind1,ind2].set_axis_off() #Don't show the axis numbers.
    axs[ind1,ind2].plot([8e3,9e3],[-1e3,-1e3],'k') #Scale bar
    axs[ind1,ind2].text(9e3,-1e3,'1 km',horizontalalignment='left',verticalalignment='center',fontsize='small')
    
cax.invert_xaxis()
cax.text(0.02, -1.2, 'Older', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)
cax.text(0.98, -1.2, 'Younger', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)

#Save the figure.
fig.savefig('Fig7.pdf',bbox_inches='tight')
fig.savefig('Fig7.png',bbox_inches='tight')