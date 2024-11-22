# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:50:05 2023

@author: oakley

Make a series of plots showing the process of clustering-based fold identification for a single map.

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

# fig = plt.figure(figsize=(9, 9),dpi=300)
fig = plt.figure(figsize=(12, 12),dpi=300)
axs = fig.subplots(nrows=3,ncols=3)
plt.subplots_adjust(hspace=0.1)
ax1,ax2,ax3 = (axs[0,0],axs[0,1],axs[0,2])
ax4,ax5,ax6 = (axs[1,0],axs[1,1],axs[1,2])
ax7,ax8,ax9 = (axs[2,0],axs[2,1],axs[2,2])

map_name = 'DoubleFold'
map_folder = '../Synthetic_Maps/'
results_folder = '../Unsupervised_Clustering_Method/tmp/'
nodes = 20 #Number of nodes for the rays, since this is used to form the file name.
use_low_dip_mask = True
use_dip_direction_mask = True

# map_file = map_folder + map_name + '.shp'
map_file = map_folder + '/' + map_name + '/' + map_name + '.shp'
rays_file = results_folder + str(nodes) + '_nodes_' + map_name + '_rays.shp'
segments_file = results_folder + str(nodes) + '_nodes_' + map_name + '_segments.shp'
match_lines_file = results_folder + str(nodes) + '_nodes_' + map_name + '_Match_lines.shp'
match_points_file = results_folder + str(nodes) + '_nodes_' + map_name + '_Match_points.shp'
cluster_points_file = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_points.shp'
cluster_lines_file = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_lines.shp'
cluster_points_file_preLOF = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_points_preLOF.shp'
cluster_lines_file_preLOF = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_lines_preLOF.shp'
fold_axes_file = results_folder + str(nodes) + '_nodes_' + map_name + '_fold_axes.shp'
fold_areas_file = results_folder + str(nodes) + '_nodes_' + map_name + '_fold_area.shp'

fold_type_colors = ['blue','red']
cluster_colors = ['blue','orange','red']

strat_num_shift = 1 #Shift down by this so the youngest unit is 1.
vmin = 1#
vmax = 7-strat_num_shift

#Define a function to plot the base map, since we will be doing this repeatedly.
def plot_map(map_file,ax,legend=True):
    map_gdf = gp.GeoDataFrame.from_file(map_file, geom_col='geometry', crs='epsg:2154')
    map_gdf['RelAge'] -= strat_num_shift
    divider = make_axes_locatable(ax)
    if legend:
        cax = divider.append_axes("right", size="5%", pad=0.1)
    else:
        cax = None
    colormap = 'Greys'
    legend_kwds = {'orientation':'vertical',
                   'label':'Relative Age',}
    map_gdf.plot(ax=ax, column='RelAge', edgecolor='black',cmap=colormap,cax=cax,
             categorical=False,legend=legend,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
    if legend:
        cax.invert_yaxis()
        cax.text(1.5, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
        cax.text(1.5, 0.07, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
    ax.set_axis_off() #Don't show the axis numbers.
    ax.plot([1.05e4,1.15e4],[1e3,1e3],'k') #Scale bar
    ax.text(1.05e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')

#Plot the map with the ray divided into segments.
plot_ray_ind = 39
plot_map(map_file,ax1,legend=True)
segments_gdf = gp.GeoDataFrame.from_file(segments_file, geom_col='geometry', crs='epsg:2154')
segs = segments_gdf[segments_gdf.index_righ==plot_ray_ind]
for i in segs.index:
    x,y = segs.geometry[i].xy
    ax1.plot(x,y,'b-')
    for j in range(len(x)):
        ax1.plot([x[j],x[j]],[y[j]-200,y[j]+200],'b-')
ax1.text(-200,1.3e4,'A',fontweight='bold',fontsize='x-large')

#Plot a map with just the segments identified as containing possible folds.
match_lines_gdf = gp.GeoDataFrame.from_file(match_lines_file, geom_col='geometry', crs='epsg:2154')
plot_map(map_file,ax2,legend=False)
for i in match_lines_gdf.index:
    x,y = match_lines_gdf.geometry[i].xy
    ax2.plot(x,y,'b-',linewidth=0.5)
ax2.text(-200,1.3e4,'B',fontweight='bold',fontsize='x-large')
    
#Plot the midpoints classified as anticlines or synclines.
plot_map(map_file,ax3,legend=False)
match_points_gdf = gp.GeoDataFrame.from_file(match_points_file, geom_col='geometry', crs='epsg:2154')
handles = [None,None]
for i,ftype in enumerate(['anticline','syncline']):
    gdf = match_points_gdf[match_points_gdf['fold_type']==ftype]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax3.plot(x,y,'.',color=fold_type_colors[i])[0]
ax3.legend(handles=handles,labels=['possible anticlines','possible synclines'])
ax3.text(-200,1.3e4,'C',fontweight='bold',fontsize='x-large')

#Plot the same thing, but with the ones that fail the topographic tests masked out.
plot_map(map_file,ax4,legend=False)   
n1x,n1y,n1z = [match_points_gdf['n1x'].values,match_points_gdf['n1y'].values,match_points_gdf['n1z'].values]
n2x,n2y,n2z = [match_points_gdf['n2x'].values,match_points_gdf['n2y'].values,match_points_gdf['n2z'].values]
nh1 = np.sqrt(n1x**2.+n1y**2.)
nh2 = np.sqrt(n2x**2.+n2y**2.)
phi1 = np.arctan(np.abs(n1z)/nh1) #Plunge
phi2 = np.arctan(np.abs(n2z)/nh2) #Plunge
dip1 = 90-phi1*180.0/np.pi
dip2 = 90-phi2*180.0/np.pi
mask = np.ones(phi1.shape,dtype=bool)
if use_low_dip_mask:
    mask = mask & (dip1>5) & (dip2>5)
dip_direction_mask = ((match_points_gdf['fold_type']=='anticline') & (match_points_gdf['DipDirs']=='apart')) | (
    (match_points_gdf['fold_type']=='syncline') & (match_points_gdf['DipDirs']=='together'))
if use_dip_direction_mask:
    mask = mask & dip_direction_mask #Use this.
handles = [None,None,None]
gdf = match_points_gdf[~mask]
x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
handles[2] = ax4.plot(x,y,'.',color='black',alpha=0.1)[0]
for i,ftype in enumerate(['anticline','syncline']):
    gdf = match_points_gdf[(match_points_gdf['fold_type']==ftype) & mask]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax4.plot(x,y,'.',color=fold_type_colors[i])[0]
ax4.legend(handles=handles,labels=['possible anticlines','possible synclines','rejected'])
ax4.text(-200,1.3e4,'D',fontweight='bold',fontsize='x-large')

#Plot the clustered midpoints.
cluster_points_gdf = gp.GeoDataFrame.from_file(cluster_points_file, geom_col='geometry', crs='epsg:2154')
gdf_sans_noise_points = cluster_points_gdf[cluster_points_gdf['cluster'] != -1]
noise = cluster_points_gdf[cluster_points_gdf['cluster'] == -1]
plot_map(map_file,ax5,legend=False)
clusters_unique = pd.unique(gdf_sans_noise_points['cluster']).tolist()
handles = [None,None,None,None]
if not noise.empty:
    x = [noise.geometry[i].xy[0][0] for i in noise.index]
    y = [noise.geometry[i].xy[1][0] for i in noise.index]
    handles[3] = ax5.plot(x,y,'.',color='black',alpha=0.1)[0]
i = 0
for color, (index, group), c in zip(cluster_colors[0:len(clusters_unique)], gdf_sans_noise_points.groupby(['cluster']), clusters_unique):
    x = [group.geometry[i].xy[0][0] for i in group.index]
    y = [group.geometry[i].xy[1][0] for i in group.index]
    handles[i] = ax5.plot(x,y,'.',color=cluster_colors[i])[0]
    i += 1
ax5.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3','not clustered'],ncols=2)
ax5.text(-200,1.3e4,'E',fontweight='bold',fontsize='x-large')

#Plot the midpoints that will be used for determining the axes.
plot_map(map_file,ax6,legend=False)
handles = [None,None,None]
for i,c in enumerate(clusters_unique):
    gdf = gdf_sans_noise_points[(gdf_sans_noise_points['cluster']==c) & (gdf_sans_noise_points['max_angle']<60)]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax6.plot(x,y,'.',color=cluster_colors[i])[0]
ax6.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3'],ncols=2)
ax6.text(-200,1.3e4,'F',fontweight='bold',fontsize='x-large')

#Plot the fold axes.
plot_map(map_file,ax7,legend=False)
axes_gdf = gp.GeoDataFrame.from_file(fold_axes_file, geom_col='geometry', crs='epsg:2154')
for i,c in enumerate(clusters_unique):
    ind = axes_gdf.index[axes_gdf.cluster==c][0] #This should only return one item.
    x,y = axes_gdf.geometry[ind].xy
    ax7.plot(x,y,'-',color=cluster_colors[i])
ax7.text(-200,1.3e4,'G',fontweight='bold',fontsize='x-large')

#Plot the line segments that will be used for determining the fold areas.
cluster_lines_gdf = gp.GeoDataFrame.from_file(cluster_lines_file, geom_col='geometry', crs='epsg:2154')
plot_map(map_file,ax8,legend=False)
handles = [None,None,None]
for i,c in enumerate(clusters_unique):
    gdf = cluster_lines_gdf[cluster_lines_gdf['cluster']==c]
    for j,ind in enumerate(gdf.index):
        x,y = gdf.geometry[ind].xy
        h = ax8.plot(x,y,color=cluster_colors[i],linewidth=0.5)[0]
        if j == 1:
            handles[i] = h
ax8.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3'],ncols=2)
ax8.text(-200,1.3e4,'H',fontweight='bold',fontsize='x-large')

#Plot the fold areas.
plot_map(map_file,ax9,legend=False)
areas_gdf = gp.GeoDataFrame.from_file(fold_areas_file, geom_col='geometry', crs='epsg:2154')
newcmp = ListedColormap(cluster_colors)
areas_gdf.plot(ax=ax9,categorical=True,cmap=newcmp,legend=False, alpha=0.5, edgecolor='black')
ax9.text(-200,1.3e4,'I',fontweight='bold',fontsize='x-large')

#Save the figure.
fig.savefig('Fig3.pdf',bbox_inches='tight')
fig.savefig('Fig3.png',bbox_inches='tight')