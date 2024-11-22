# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:54:53 2024

@author: david

Prepare raster ata of the Esternay map for interpretation with UNET.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

#Produce an npz file with the raster relative ages and DEM.
name = 'Lavelanet'

#Read in the raster map data.
map_file_path = 'Lavelanet_Raster.tif'
geo_map = rasterio.open(map_file_path)
map_shape = geo_map.shape #It looks like this gives (rows,columns)

#Read in the elevation data.
dem_file_path = './Lavelanet_DEM.tif'
dem = rasterio.open(dem_file_path)

#Get the xy coordinates of the grid points of the geological map.
cols, rows = np.meshgrid(np.arange(map_shape[1]), (map_shape[0]-1)-np.arange(map_shape[0])) #The rows and columns are numbered from top left.
xs, ys = geo_map.xy(rows, cols)
x = np.array(xs)
y = np.array(ys)

#Get the DEM elevations at the grid points of the geological map.
z = np.zeros(map_shape)
g = np.zeros(map_shape,dtype=int)
for i in range(z.shape[0]):
    coord_list = [(x,y) for x,y in zip(x[i,:],y[i,:])]
    zp = [x[0] for x in dem.sample(coord_list)]
    z[i,:] = zp
    gs = [x[0] for x in geo_map.sample(coord_list)]
    g[i,:] = gs
    
#In areas with no geologic map data, set g to 0.
#We are not (yet) going to set z to 0 in the same place, because that would mess up the scaling.
nodata = g==geo_map.nodata
nodata[z == 0] = True #If z does not get set in some places, set those places to no data too.
g[nodata] = 0 #Other no data values may have been used before.

#Plot the cropped model area and elevation.
plt.figure()
plt.imshow(z,origin='lower')
plt.title(name + ' Elevation')
plt.axis('Equal')
plt.figure()
plt.imshow(g,origin='lower')
plt.title(name + ' Geology')
plt.axis('Equal')

#Save the model.
np.savez(name,unit=g,topo=z,nodata=nodata)

