# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:26:25 2023

@author: oakley

This is a function to create an uplift pattern for a fold consisting purely of vertical uplift.
"""

import numpy as np
from shapely import LineString, Polygon

def fold(Xin,Yin,cell_size,xc,yc,amplitude,wavelength,vergence,plunge_wavelength=0,asymmetry=1,sigmoid_amplitude=0,sigmoid_steepness=1,sigmoid_scale=None):
    """
    
    Parameters
    ----------
    Xin : numpy array
        x coordinates of points at which uplift should be calculated. (Must be a regular grid in meshgrid format.)
    Yin : numpy array
        y coordinates of points at which uplift should be calculated (Must be a regular grid in meshgrid format.)
    cell_size : float
        Size of the grid cells for Xin and Yin.
    xc: float
        x coordinate of the highest point of the fold (center of fold axis)
    yc: float
        y coordinate of the highest point of the fold (center of fold axis)
    amplitude : float
        Fold amplitude
    wavelength : float
        Fold wavelength (from forelimb syncline to backlimb syncline)
    vergence : float
        Fold vergence direction, in degrees, specified as a Cartesian angle from the x-axis (E), not as an azimuth.
    plunge_wavelength : float
        Wavelength in the fold axial direction, creating a doubly plunging fold. Set to 0 for a non-plunging fold.
    asymmetry: float
        Ratio of the forelimb wavelength to the backlimb wavelength. Setting to 0 (as well as 1) will make it not asymmetric.
    sigmoid_amplitude : float
        Amplitude of the logistic function used to make a sigmoidal fold axis. Set it to 0 to not have a sigmoid.
    sigmoid_steepness : float
        The steepness value (k) of the logistic function used to make a sigmoidal fold axis. Smaller values give gentler curves.
        Positive values offset top to the right, and negative values top to the left, in the rotated coordinate system.
    sigmoid_scale : float
        A value by which to divide the along-axis distance when calculating the logistic function. The default is plunge_wavelength/10.

    Returns
    -------
    uplift : numpy array
        The amount of uplift at each point in (Xin,Yin)
        
    """
    
    if np.any(np.diff(Xin,axis=0) != cell_size) or np.any(np.diff(Yin,axis=1) != cell_size):
        print('Error: Grid must be in meshgrid format.')
    
    #Rotate to work in a coordinate system with the fold vergence direction being the x axis.
    v = vergence*np.pi/180 #Convert to radians.
    X = (Xin-xc)*np.cos(-v)-(Yin-yc)*np.sin(-v)
    Y = (Xin-xc)*np.sin(-v)+(Yin-yc)*np.cos(-v)
    xcf = 0.0
    ycf = 0.0
    xwavelength = wavelength
    ywavelength = plunge_wavelength
    
    #Make a line for the fold axis.
    axis_y = ycf+np.linspace(-plunge_wavelength/2,plunge_wavelength/2,num=int(plunge_wavelength/cell_size+1))
    axis_x = xcf*np.zeros(axis_y.shape)
    
    #Determine some things about the fold.
    plunging = plunge_wavelength != 0
    asymmetric = (asymmetry != 0) & (asymmetry != 1)
    sigmoidal = sigmoid_amplitude != 0
    
    #Deal with asymmetry.
    if asymmetric:
        #The wavelengths are related by:
            #   wavelength1/wavelength2 = asymmetry
            #   wavelength1/2 + wavelength2/2 = xwavelength
        wavelength1 = 2.*xwavelength/(1.+1./asymmetry) #Short wavelength
        wavelength2 = 2.*xwavelength/(1.+asymmetry) #Long wavelength
    else:
        wavelength1,wavelength2 = [xwavelength,xwavelength]
    xcf_eff = xcf-xwavelength/2.+wavelength2/2. #Effective xcf (x coordinate of fold axis).
    axis_x = axis_x-xwavelength/2.+wavelength2/2.
    
    #Make a polygon for the fold area.
    t = np.linspace(0.0,2*np.pi,num=360)
    x1,y1 = [xcf+(wavelength1/2.)*np.cos(t),ycf+(ywavelength/2)*np.sin(t)]
    x2,y2 = [xcf+(wavelength2/2.)*np.cos(t),ycf+(ywavelength/2)*np.sin(t)]
    backlimb_mask = np.cos(t)<=0.0
    area_x,area_y = [x1,y1]
    area_x[backlimb_mask] = x2[backlimb_mask]
    area_y[backlimb_mask] = y2[backlimb_mask]
    area_x = area_x-xwavelength/2.+wavelength2/2. #Adjust for asymmetry.
    
    #Deal with a sigmoidal shape.
    if sigmoidal:
        if sigmoid_scale is None:
            sigmoid_scale = plunge_wavelength/10.0
        scaled_dist = (Y-ycf)/sigmoid_scale #Scaled along axis distance from the center of the fold.
        logistic_fun = sigmoid_amplitude/(1.+np.exp(-sigmoid_steepness*scaled_dist))
        xcf_eff = xcf_eff+(logistic_fun-sigmoid_amplitude/2) #Center the offset, since logistic function itself goes from 0 to sigmoid_amplitude.
        axis_scaled_dist = (axis_y-ycf)/sigmoid_scale
        axis_logistic_fun = sigmoid_amplitude/(1.+np.exp(-sigmoid_steepness*axis_scaled_dist))
        axis_x = axis_x+(axis_logistic_fun-sigmoid_amplitude/2)
        area_scaled_dist = (area_y-ycf)/sigmoid_scale
        area_logistic_fun = sigmoid_amplitude/(1.+np.exp(-sigmoid_steepness*area_scaled_dist))
        area_x = area_x+(area_logistic_fun-sigmoid_amplitude/2)
    
    #Calculate the uplift.
    #uplift is amplitude times a cosine curve, which is scaled and shifted to go from 0 to 1 instead of -1 to 1.
    uplift = np.zeros(X.shape)
    if not plunging:
        r1 = (X-xcf_eff)/(wavelength1/2.)
        r2 = (X-xcf_eff)/(wavelength2/2.)
        uplift1 = amplitude*((1.0+np.cos(2.0*np.pi*(X-xcf_eff)/wavelength1))/2.0)
        uplift2 = amplitude*((1.0+np.cos(2.0*np.pi*(X-xcf_eff)/wavelength2))/2.0)
    else:
        r1 = np.sqrt(((X-xcf_eff)/(wavelength1/2.))**2.+((Y-ycf)/(ywavelength/2.))**2.)
        r2 = np.sqrt(((X-xcf_eff)/(wavelength2/2.))**2.+((Y-ycf)/(ywavelength/2.))**2.)
        uplift1 = amplitude*((1.0+np.cos(2.0*np.pi*(X-xcf_eff)/wavelength1))/2.0)*((1+np.cos(2.0*np.pi*(Y-ycf)/ywavelength))/2.0)
        uplift2 = amplitude*((1.0+np.cos(2.0*np.pi*(X-xcf_eff)/wavelength2))/2.0)*((1+np.cos(2.0*np.pi*(Y-ycf)/ywavelength))/2.0)
    mask1 = (r1<=1.0) & (X>=xcf_eff)
    mask2 = (r2<=1.0) & (X<=xcf_eff)
    uplift[mask1] = uplift1[mask1]
    uplift[mask2] = uplift2[mask2]
    fold_area = mask1 | mask2
    
    #Determine whether the fold axis passes through a grid square or not.
    if not sigmoidal:
        xcf_eff = xcf_eff*np.ones(X.shape) #xcf needs to be a grid.
    axis_in_cell_x = (X-cell_size/2. < xcf_eff) & (X+cell_size/2. >= xcf_eff)
    if plunging:
        axis_in_cell_y = (Y >= ycf-ywavelength/2.0) & (Y <= ycf+ywavelength/2.0)
        axis_in_cell = axis_in_cell_x & axis_in_cell_y
    else:
        axis_in_cell = axis_in_cell_x
    
    #Rotate axis and area back:
    axis_x_rot = axis_x*np.cos(v)-axis_y*np.sin(v) + xc
    axis_y_rot = axis_x*np.sin(v)+axis_y*np.cos(v) + yc
    area_x_rot = area_x*np.cos(v)-area_y*np.sin(v) + xc
    area_y_rot = area_x*np.sin(v)+area_y*np.cos(v) + yc
    
    #Create shapely objects for the fold axis and fold area.
    fold_axis = LineString([(axis_x_rot[i],axis_y_rot[i]) for i in range(len(axis_x_rot))])
    fold_area_poly = Polygon([(area_x_rot[i],area_y_rot[i]) for i in range(len(area_x_rot))])
    
    return uplift, axis_in_cell, fold_area, fold_axis, fold_area_poly