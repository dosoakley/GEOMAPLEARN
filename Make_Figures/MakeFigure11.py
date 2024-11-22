# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:58:19 2023

@author: oakley

Plot the real-world maps re-colored by relative age.
"""

import geopandas as gp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.gridspec as gridspec
import rasterio

fig = plt.figure(figsize=(9, 8),dpi=300)
gs1 = gridspec.GridSpec(2,2,height_ratios=[0.4,0.6],width_ratios=[0.55,0.46],bottom=0.4) #These ratios get the 1 km scale bars in both maps approximately the same size.
gs1.update(hspace=0.0)
gs2 = gridspec.GridSpec(1,1,top=0.4,left=0.14,right=0.46)
gs3 = gridspec.GridSpec(1,1,top=0.4,left=0.59,right=0.93)
ax1 = fig.add_subplot(gs1[0,0]) #Axis for the location map.
ax2 = fig.add_subplot(gs1[1,0]) #Axis for the Lavelanet map.
ax3 = fig.add_subplot(gs1[0:2,1]) #Axis for the Esternay map.
ax4 = fig.add_subplot(gs2[0,0]) #Axis for the Lavelanet topographic map.
ax5 = fig.add_subplot(gs3[0,0]) #Axis for the Esternay topographic map.

map_file_Lavelanet = '../Real_Maps/Lavelanet/Lavelanet_map.shp'
map_file_Esternay = '../Real_Maps/Esternay/Esternay_map.shp'

#Load and plot the location map as subfigure A.
image = plt.imread('./LocationMap.png')
im = ax1.imshow(image)
ax1.set_aspect('equal',adjustable='box')
ax1.set_axis_off() #Don't show the axis numbers.
ax1.text(0.0,1.05,'A',fontsize='large',fontweight='bold',transform=ax1.transAxes)

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

#Define some settings for the maps.
cmap = 'viridis_r'
edgecolor = 'black'
linewidth = 0.1

#Plot the Lavelanet map as figure B.
map_file = map_file_Lavelanet
map_gdf1 = gp.GeoDataFrame.from_file(map_file, geom_col='geometry', crs='epsg:2154')
map_gdf1.RelAge = ShiftAges(map_gdf1.RelAge.values)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
colormap = cmap
legend_kwds = {'orientation':'vertical',
               'label':'Relative Age'}
vmax = map_gdf1['RelAge'].max()
map_gdf1.plot(ax=ax2, column='RelAge', edgecolor=edgecolor,cmap=colormap,cax=cax,
         categorical=False,legend=True,vmin=0,vmax=vmax,linewidth=linewidth,
         legend_kwds=legend_kwds)
cax.invert_yaxis()
cax.text(0.5, 1.09, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
cax.text(0.5, -0.1, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
minx,miny,maxx,maxy = map_gdf1.total_bounds
xbar = minx + (maxx-minx)*0.92
ybar = miny + (maxy-miny)*0.05
ax2.plot([xbar,xbar+1e3],[ybar,ybar],'k') #Scale bar
xrange = maxx-minx
ax2.text(xbar-1e3,ybar+100,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax2.set_axis_off() #Don't show the axis numbers.
ax2.text(0.05,1.02,'B',fontsize='large',fontweight='bold',transform=ax2.transAxes)

#Plot the Esternay map as figure C.
map_file = map_file_Esternay
map_gdf2 = gp.GeoDataFrame.from_file(map_file, geom_col='geometry', crs='epsg:2154')
map_gdf2.RelAge = ShiftAges(map_gdf2.RelAge.values)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
colormap = cmap
legend_kwds = {'orientation':'vertical',
               'label':'Relative Age'}
vmax = map_gdf2['RelAge'].max()
map_gdf2.plot(ax=ax3, column='RelAge', edgecolor=edgecolor,cmap=colormap,cax=cax,
         categorical=False,legend=True,vmin=0,vmax=vmax,linewidth=linewidth,
         legend_kwds=legend_kwds)
cax.invert_yaxis()
cax.text(0.5, 1.06, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
cax.text(0.5, -0.05, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
minx,miny,maxx,maxy = map_gdf2.total_bounds
xbar = minx + (maxx-minx)*0.92
ybar = miny + (maxy-miny)*0.05
ax3.plot([xbar,xbar+1e3],[ybar,ybar],'k') #Scale bar
xrange = maxx-minx
ax3.text(xbar-1e3,ybar+100,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax3.set_axis_off() #Don't show the axis numbers.
ax3.text(0.05,0.97,'C',fontsize='large',fontweight='bold',transform=ax3.transAxes)

#Plot the topography for Lavelanet.
dem_file_path = '../Real_Maps/Lavelanet/Lavelanet_DEM.tif'
dem1 = rasterio.open(dem_file_path)
dem_shape = dem1.shape #It looks like this gives (rows,columns)
topo1 = dem1.read(1).astype(float) #1 is the channel number. There is only 1 channel in this image.
im = ax4.imshow(topo1,cmap='cividis',extent=[dem1.bounds.left,dem1.bounds.right,dem1.bounds.bottom,dem1.bounds.top])
minx,miny,maxx,maxy = map_gdf1.total_bounds
ax4.set_xlim(minx,maxx) 
ax4.set_ylim(miny,maxy)
ax4.set_xticks([])
ax4.set_yticks([])
cbar4 = plt.colorbar(im,ax=ax4,orientation='horizontal')
cbar4.set_label('Elevation (m)')
ax4.text(0.0,1.04,'D',fontsize='large',fontweight='bold',transform=ax4.transAxes)

#Plot the topography for Esternay.
dem_file_path = '../Real_Maps/Esternay/Esternay_DEM.tif'
dem2 = rasterio.open(dem_file_path)
dem_shape = dem2.shape #It looks like this gives (rows,columns)
topo2 = dem2.read(1).astype(float) #1 is the channel number. There is only 1 channel in this image.
im = ax5.imshow(topo2,cmap='cividis',extent=[dem2.bounds.left,dem2.bounds.right,dem2.bounds.bottom,dem2.bounds.top])
minx,miny,maxx,maxy = map_gdf2.total_bounds
ax5.set_xlim(minx,maxx) 
ax5.set_ylim(miny,maxy)
ax5.set_xticks([])
ax5.set_yticks([])
cbar5 = plt.colorbar(im,ax=ax5,orientation='vertical')
cbar5.set_label('Elevation (m)')
ax5.text(0.0,1.03,'E',fontsize='large',fontweight='bold',transform=ax5.transAxes)

#Save the figure.
fig.savefig('Fig11.pdf',bbox_inches='tight')
fig.savefig('Fig11.png',bbox_inches='tight')