# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 00:26:22 2023

@author: david

Plot the synthetic model map, topography, and cross section.
"""

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

#See here: https://matplotlib.org/2.0.2/users/gridspec.html for how to create a layout with different sized plots.

#Create a figure with the required number of subplots.
fig = plt.figure(figsize=(9, 6),dpi=300)
gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.1,1,1.1]) #For some reason, the plots made with geopandas plot come out a bit smaller than the middle one.
gs1.update(bottom=0.4)
ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[0,1])
ax3 = fig.add_subplot(gs1[0,2])
gs2 = gridspec.GridSpec(1,1)
gs2.update(top=0.4,left=0.2,right=0.8)
ax4 = fig.add_subplot(gs2[0])
cax = fig.add_axes([0.83, 0.12, 0.02, 0.28]) #This is where  the colorbar will be plotted.

map_folder = '../Synthetic_Maps/'
map_file_no_topo = map_folder + 'DoubleFold_NoTopo/DoubleFold_NoTopo.shp'
map_file_topo = map_folder + 'DoubleFold/DoubleFold.shp'
xsec_file = map_folder + 'DoubleFold/DoubleFold_cross_section.shp'
xsec_topo_profile = map_folder + 'DoubleFold/DoubleFold_topo.npz'
axes_file = map_folder+'DoubleFold/DoubleFold_anticline_axes.shp'
npzfile = np.load(xsec_topo_profile)

colormap = 'viridis_r'
strat_num_shift = 1 #Shift down by this so the youngest unit is 1.
vmin = 1#0
vmax = 7-strat_num_shift
legend_kwds = {'orientation':'vertical',
               'ticks':np.arange(vmin,vmax+1)}

#Plot the map without topography.
map_gdf = gp.GeoDataFrame.from_file(map_file_no_topo, geom_col='geometry', crs='epsg:2154')
map_gdf['RelAge'] -= strat_num_shift
map_gdf.plot(ax=ax1, column='RelAge', edgecolor='black',cmap=colormap,
         categorical=False,legend=False,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
ax1.set_axis_off() #Don't show the axis numbers.
ax1.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax1.text(1.05e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax1.text(200,1.15e4,'A',fontweight='bold',fontsize='large')
#Plot the fold axes.
axes_gdf = gp.GeoDataFrame.from_file(axes_file, geom_col='geometry', crs='epsg:2154')
for i in axes_gdf.index:
    x,y = axes_gdf.geometry[i].xy
    x,y = [np.array(x),np.array(y)]
    mask = y >= 0.0
    x,y = [x[mask],y[mask]]
    ax1.plot(x,y,'k:')

#Plot the topography.
X = npzfile['X']
Y = npzfile['Y']
topo = npzfile['topo']
levels = np.arange(start=3000,stop=3800,step=100)
cs = ax2.contour(X,Y,topo,levels,cmap='gist_earth')
ax2.clabel(cs, inline=True, fontsize=8)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax2.text(1.05e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax2.set_aspect('equal',adjustable='box') #For the ones plotted with geopandas, this is already true.
ax2.plot([0,12760,12760,0,0],[0,0,12760,12760,0],'-k')
ax2.text(200,1.15e4,'B',fontweight='bold',fontsize='large')

#Plot the map with topography.
map_gdf2 = gp.GeoDataFrame.from_file(map_file_topo, geom_col='geometry', crs='epsg:2154')
map_gdf2['RelAge'] -= strat_num_shift
map_gdf2.plot(ax=ax3, column='RelAge', edgecolor='black',cmap=colormap,
          categorical=False,legend=False,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
ax3.set_axis_off() #Don't show the axis numbers.
ax3.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax3.text(1.05e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax3.text(200,1.15e4,'C',fontweight='bold',fontsize='large')
#Plot topographic contours.
cs3 = ax3.contour(X,Y,topo,levels,colors='gray',alpha=0.5)
ax3.clabel(cs3, inline=True, fontsize=6)
xlims,ylims = [ax1.get_xlim(),ax1.get_ylim()] #The contouring can change the figure size, so use these to change it back.
ax3.set_xlim(xlims[0],xlims[1]) 
ax3.set_ylim(ylims[0],ylims[1])
# Add the cross section line to the map with topography plot.
[cs_start,cs_end] = npzfile['cs_line']
ax3.plot([cs_start[0],cs_end[0]],[cs_start[1],cs_end[1]],'-k')
ax3.text(cs_start[0]-200,cs_start[1],'A',horizontalalignment='right',verticalalignment='center')
ax3.text(cs_end[0]+200,cs_end[1],'A′',horizontalalignment='left',verticalalignment='center')

#Plot the cross section.
xsec_gdf = gp.GeoDataFrame.from_file(xsec_file, geom_col='geometry', crs='epsg:2154')
xsec_gdf['RelAge'] -= strat_num_shift
xsec_gdf.plot(ax=ax4, column='RelAge', edgecolor='black',cmap=colormap,cax=cax,
         categorical=False,legend=True,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Elevation (m)')
ax4.text(-500,3.4e3,'D',fontweight='bold',fontsize='large')
ax4.text(0,4.0e3,'A',horizontalalignment='center')
cs_length = np.sqrt((cs_end[0]-cs_start[0])**2 + (cs_end[1]-cs_start[1])**2)
ax4.text(cs_length,4.0e3,'A′',horizontalalignment='center')

#Label the colorbar.
cax.invert_yaxis() #Make it go older to younger.
cax.set_yticklabels(['1 (younger)','2','3','4','5','6 (older)'])
cax.text(2.5,3.5,'Relative Age',rotation='vertical',verticalalignment='center',horizontalalignment='left')

#Save the figure.
fig.savefig('Fig2.pdf',bbox_inches='tight')
fig.savefig('Fig2.png',bbox_inches='tight')
