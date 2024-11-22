# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:14:35 2023

Plot the clustering method results for the real-world maps.

@author: oakley
Useful references for plotting with geopandas:
    https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.plot.html
    https://geopandas.org/en/stable/docs/user_guide/mapping.html
"""

import geopandas as gp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 9),dpi=300)
gs = gridspec.GridSpec(2,2,width_ratios=[0.55,0.46])
gs.update(hspace=0.0)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[0,1])
ax4 = fig.add_subplot(gs[1,1])
axs = [ax1,ax2,ax3,ax4]

map_file_Lavelanet = '../Real_Maps/Lavelanet/Lavelanet_map.shp'
map_file_Esternay = '../Real_Maps/Esternay/Esternay_map.shp'
map_files = [map_file_Lavelanet,map_file_Esternay]
names = ['Lavelanet','Esternay']

#Define a function to plot the base map, since we will be doing this repeatedly.
def plot_map(map_gdf,ax,legend=True,cmap='Greys',edgecolor='black',linewidth=1.0):
    divider = make_axes_locatable(ax)
    if legend:
        cax = divider.append_axes("right", size="5%", pad=0.1)
    else:
        cax = None
    colormap = cmap
    legend_kwds = {'orientation':'vertical',
                   'label':'Relative Age'}
    vmax = map_gdf['RelAge'].max()
    map_gdf.plot(ax=ax, column='RelAge', edgecolor=edgecolor,cmap=colormap,cax=cax,
             categorical=False,legend=legend,vmin=0,vmax=vmax,linewidth=linewidth,
             legend_kwds=legend_kwds)
    if legend:
        cax.invert_yaxis()
        cax.text(1.8, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
        cax.text(1.8, 0.05, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
    minx,miny,maxx,maxy = map_gdf.total_bounds
    ax.set_axis_off() #Don't show the axis numbers.
    xbar = minx + (maxx-minx)*0.9
    ybar = miny + (maxy-miny)*0.05
    ax.plot([xbar,xbar+1e3],[ybar,ybar],'k') #Scale bar
    ax.text(xbar,ybar+100,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')

#Define a function to shift the ages so there aren't any gaps, and 0 is always Quaternary.
def ShiftAges(ages):
    ages_unique = np.unique(ages)
    if not np.any(ages_unique==0):
        ages_unique = np.concatenate([0],ages_unique)
    for j in range(1,len(ages_unique)):
        shift = ages_unique[j] - ages_unique[j-1] - 1
        if shift > 0: #There's a gap.
            ages[ages >= ages_unique[j]] -= shift
            ages_unique[j:] -= shift
    return ages

ax_ind = 0
letters = ['A','C','B','D']
bbox_props = dict(boxstyle='square', facecolor='white', alpha=1.0)
for i in range(len(map_files)):
    #Get the files.
    map_file = map_files[i]
    map_gdf = gp.GeoDataFrame.from_file(map_file, geom_col='geometry', crs='epsg:2154')
    map_gdf.RelAge = ShiftAges(map_gdf.RelAge.values)
    fold_axes_file = '../Unsupervised_Clustering_Method/tmp/20_nodes_'+names[i]+'_fold_axes.shp'
    axes_gdf = gp.GeoDataFrame.from_file(fold_axes_file, geom_col='geometry', crs='epsg:2154')
    fold_areas_file = '../Unsupervised_Clustering_Method/tmp/20_nodes_'+names[i]+'_fold_area.shp'
    areas_gdf = gp.GeoDataFrame.from_file(fold_areas_file, geom_col='geometry', crs='epsg:2154')

    #Set the positioning of the subplot labels a bit differently for the two maps.
    xletter = 0.09
    if i == 0:
        yletter = 0.87
    else:
        yletter = 0.89

    #Plot the fold axes.
    plot_map(map_gdf,axs[ax_ind],legend=True,linewidth=0.1)
    clusters_unique = np.sort(pd.unique(axes_gdf['cluster'])).tolist()
    colors = ['blue','red']
    for i,c in enumerate(clusters_unique):
        ind = axes_gdf.index[axes_gdf.cluster==c][0] #This should only return one item.
        x,y = axes_gdf.geometry[ind].xy
        color = colors[axes_gdf.fold_type[ind]]
        axs[ax_ind].plot(x,y,'-',color=color,linewidth=2)
    axs[ax_ind].text(xletter,yletter,letters[ax_ind],fontsize='large',fontweight='bold',transform=axs[ax_ind].transAxes,
                     bbox=bbox_props, horizontalalignment='center',verticalalignment='center')
    ax_ind += 1

    #Plot the fold areas.
    plot_map(map_gdf,axs[ax_ind],legend=True,linewidth=0.1)
    newcmp = ListedColormap(colors)
    areas_gdf.plot(column='fold_type',ax=axs[ax_ind],categorical=True,cmap=newcmp,legend=False, alpha=0.5, edgecolor='black')
    axs[ax_ind].text(xletter,yletter,letters[ax_ind],fontsize='large',fontweight='bold',transform=axs[ax_ind].transAxes,
                     bbox=bbox_props, horizontalalignment='center',verticalalignment='center')
    ax_ind += 1

#Add labels that apply to the whole columns or rows.
axs[0].text(-0.01,0.5,'Fold Axes',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',rotation=90,transform=axs[0].transAxes)
axs[1].text(-0.01,0.5,'Fold Ares',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',rotation=90,transform=axs[1].transAxes)
axs[0].text(-0.9,1.02,'Lavelanet',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',transform=axs[2].transAxes) #Use the transform from axs[2] so it lines up vertically with the Esternay one.
axs[2].text(0.5,1.02,'Esternay',horizontalalignment='center',verticalalignment='center',
            fontweight='bold',fontsize='x-large',transform=axs[2].transAxes)

#Save the figure.
fig.savefig('Fig12.pdf',bbox_inches='tight')
fig.savefig('Fig12.png',bbox_inches='tight')