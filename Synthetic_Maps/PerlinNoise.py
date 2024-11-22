# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:46:11 2023

@author: oakley

Here, I am attempting to make a fast code for generating Perlin noise on a grid, 
using numpy vectorization and numba.

See the Wikipedia article on Perlin noise for an explanation and an example code 
that this one takes some inspiration from.
Link: https://en.wikipedia.org/wiki/Perlin_noise (accessed August 4, 2023), 

This code is made for the rapid creation of Perlin noise on a regular grid.
"""

import numpy as np
from numpy.random import rand, seed
from numba import jit

@jit(nopython=True)
def interpolate(a0, a1, w):
    """
     Interpolate with Smootherstep.
    """
    return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a0; #This looks even better.

@jit(nopython=True)
def index_spiral(Nx,Ny):
    """
    This function assigns single digit indices to the elements of a matrix in a spiral manner:
        (0,0) = 0, (1,0) = 1, (1,1) = 2, (0,1) = 3, (2,0) = 4, (2,1) = 5, (2,2) = 6, (1,2) = 7, (0,2) = 8, etc.
    This way, if the size of the matrix is increased in either or both dimensions, the index assigned to a given element will not change.
    This can, therefore, be used to sample a sequence of random numbers so that the random number assigned to a given Perlin noise grid square 
    does not change as the total number of squares changes.
    The original indices are assumed to be in column-major order, so it converts from column-major order to the spiral order.
    """
    indN = np.empty(Nx*Ny,dtype=np.int32)
    for i in range(Nx):
        for j in range(0,i): #i >= j case
            ind = i*Ny+j
            indN[ind] = i**2+j
        for j in range(i,Ny): #i < j case
            ind = i*Ny+j
            indN[ind] = (j+1)**2-(i+1)
    return indN
    

#Compute Perlin noise at coordinates x, y
@jit(nopython=True)
def perlin_grid(nx, ny, frequency, base=0, octaves=4, lacunarity=2, persistence=0.5):
    #nx ny, and frequency should be integers.
    
    #Coefficients to multiple different things by when generating the random number seed.
    #These were chosen by blindly hitting the keyboard; there is no special meaning to them.
    base_coeff = 29876
    
    #Create the points.
    #The order generated here should be the same as if we did meshgrid with ij indexing followed by ravel.
    scale = float(frequency) #To start with. It will then decrease with each octave.
    x = np.zeros((nx*ny),dtype=np.float64)
    y = np.zeros((nx*ny),dtype=np.float64)
    x0 = np.zeros((nx*ny),dtype=np.int32)
    y0 = np.zeros((nx*ny),dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            ind = i*ny+j
            x[ind] = i/scale
            y[ind] = j/scale
    grid_shape = (nx,ny)
    npts = x.size
    
    #Make some noise.
    value = np.zeros(nx*ny,dtype=np.float64)
    amplitude = 1.0 #amplitude by which to scale the noise value for each octave.
    for oc  in range(octaves):
        #We have to regenerate the x0 and y0 values, since they change when the grid size changes.
        x0[:] = np.floor(x) #Using the x0[:] syntax keeps x0 as integer type, even though np.floor returns a float.
        y0[:] = np.floor(y)

        #Generate random vectors for all grid cell corners.
        ncor_x = int(np.ceil(nx/scale))+1
        ncor_y = int(np.ceil(ny/scale))+1
        rseed = int(base*base_coeff+oc)
        seed(rseed)
        N = max(ncor_x,ncor_y)
        rn_N = rand(N*N) #Create the amount of random numbers that would be needed if we had an NxN grid.
        indN = index_spiral(ncor_x,ncor_y) #Index the actual ncor_x x ncor_y grid in a spiral pattern for an NxN grid.
        rn = rn_N[indN]*2.0*np.pi #Assign random numbers to the real grid from the NxN grid following a spiral pattern.
        
        vx = np.cos(rn)
        vy = np.sin(rn)
        
        #Loop through the 4 corners of each grid cell, and calculate the dot products of the gradient and position vectors there.
        #There is some repetition in the generation of randomGradients here, since some corners may be used more than once, 
        #but not as much as when I was calling randomGradients separately for each corner of each (x,y) point.
        n = np.zeros((2,2,npts))
        for i in range(2):
            for j in range(2):
                #Compute the coordinates of the corner and the corresponding index of the gradient vectors.
                xc,yc = [x0+i,y0+j] #i and j will be 0 or 1. Adding 1 to the left / bottom coordinate give the right / top one.
                ind = xc*ncor_y+yc
                
                #Compute the distance vector
                dx = x - xc;
                dy = y - yc;
                
                #Compute the dot product of the gradient and the distance vector.
                n[i,j,:] = dx*vx[ind] + dy*vy[ind]

        #Determine interpolation weights
        #Could also use higher order polynomial/s-curve here
        sx = x - x0
        sy = y - y0
    
        #Interpolate along the x axes.
        ix0 = interpolate(n[0,0], n[1,0], sx)
        ix1 = interpolate(n[0,1], n[1,1], sx)
    
        #Interpolate along the y axis.
        value += amplitude * interpolate(ix0, ix1, sy)
        
        #Update the scale and amplitude for the next octave.
        x *= lacunarity
        y *= lacunarity
        scale /= lacunarity
        amplitude *= persistence
    
    #Convert back to the original shape.
    value = np.reshape(value,grid_shape)

    return value #Will return in range -1 to 1. To make it in range 0 to 1, multiply by 0.5 and add 0.5