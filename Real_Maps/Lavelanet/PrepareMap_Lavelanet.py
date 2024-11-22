# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:29:57 2024

@author: david

Merge the parts of the BC Charm-50 dataset needed for the Lavelanet anticline 
and process them to produce files that can be used by GEOMAPLEARN.

This version doesn't do the dissolve and explode in order to be more like what I 
had with the Lavelanet map.

Also, this works in two parts with some manual editing in ArcGIS in between.
"""

# import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import csv
import numpy as np
import rasterio

#Read in the geologic map data.
mapdata = gpd.read_file('Lavelanet_map_clipped_merged.shp',engine='pyogrio')

#Keep track of fid before we use explode.
mapdata['fid'] = mapdata.index

#Separate multipolgyons into indvidual polygons.
#I'm not sure why we were getting multipolygons here, but it may have been due to something I did in ArcGIS.
mapdata = mapdata.explode(ignore_index=True)

#Load the relative age information.
stratigraphy = []
stratigraphy_rel_ages = []
with open('UnitRelAges_Lavelanet.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(reader):
        if i > 0: #Skip the header line.
            stratigraphy_rel_ages.append(int(row[0]))
            stratigraphy.append(row[2]) #These are the codes that correspond to those in the NOTATION column.
        
#Assign stratigraphic ordering to the units based on the NOTATION field.
notation_vals = mapdata.NOTATION.values
strat_vals = 999*np.ones(notation_vals.shape,dtype=int) #999 will be a place holder.
for i in range(len(stratigraphy)):
    mask = notation_vals==stratigraphy[i]
    strat_vals[mask] = stratigraphy_rel_ages[i]
    if mask.sum() == 0.0:
        print('Warning: No polygons founds with NOTATION = '+stratigraphy[i])

#Check for any unassigned values.
if np.any(strat_vals == 999):
    not_classified = np.argwhere(strat_vals == 999)
    for i in range(not_classified.size):
        print(notation_vals[not_classified[i]] + ' was not classified.')
        
#Add the stratigraphic order to the geo data frame.
mapdata['RelAge'] = strat_vals

#For some reason, in the western part of the map, the units that were e1 and C7b in the 
#Souquet et al. (1984) Lavelanet map have both been coded as c6b-e1-2 here, despite having 
#c6b-e1-2(1) between them in many places. To give them different relative ages, as in Souquet et al. (1984) 
#and the eastern part of this map, I am manually changin the RelAge code for the ones that correspond to C7b here.
#This has also required me to edit Lavelanet_map_clipped_merged.shp slightly, to separate the two parts of 
#the c6b-e1-2 unit where they were merged together.
C7b_fids = [52,91,48,22,32]
for fid in C7b_fids:
    mapdata.loc[list(np.where(mapdata.fid==fid)[0]),'RelAge'] = 9 #9 is Maastrichtien. Previously, these would have been 7, which is Vitrollien

#Get rid of the fid column.
#We don't need it anymore, and it may cause confusion with the index, which may be different.
mapdata.drop(columns='fid')

#Read in the elevation data.
dem_file_path = './Lavelanet_DEM.tif'
dem = rasterio.open(dem_file_path)

#Get elevations for all points that make up each polygon.
polygons = mapdata['geometry'].values
new_polygons = []
for ip,p in enumerate(polygons):
    ext = p.exterior
    coords = list(ext.coords)
    xp,yp = [[c[0] for c in coords],[c[1] for c in coords]]
    coord_list = [(x,y) for x,y in zip(xp,yp)]
    zp = [x[0] for x in dem.sample(coord_list)]
    ext_new = [coords for coords in zip(xp, yp, zp)]

    intr = p.interiors
    intr_new = []
    for j,ring in enumerate(p.interiors):
        coords = list(ring.coords)
        xp,yp = [[c[0] for c in coords],[c[1] for c in coords]]
        coord_list = [(x,y) for x,y in zip(xp,yp)]
        zp = [x[0] for x in dem.sample(coord_list)]
        intr_new.append([coords for coords in zip(xp, yp, zp)])
        # print(np.sum(zp == -9999.0)/len(zp))
    
    new_polygons.append(Polygon(ext_new,intr_new))

#Replace the old polygons with the new ones that have elevation.
mapdata['geometry'] = new_polygons

#Save the map.
mapdata.to_file('Lavelanet_map.shp',engine='pyogrio')
