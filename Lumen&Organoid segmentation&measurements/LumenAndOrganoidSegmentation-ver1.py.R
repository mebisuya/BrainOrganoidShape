#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:30:04 2025

@author: marchenk
"""

#Code written by Marina Marchenko and Guillermo Martínez-Ara
###  This is a pipeline to segment apical signal in lumens of cerebral organoids from data obtained in lightsheet.

#The aim is to segment the lumens in 3D and save 3D segmented file

from skimage import io, filters, measure, morphology, transform
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import math
from os import listdir, mkdir
from os.path import join
from tifffile import imwrite

# --------------------------------------------
# USER-DEFINED PARAMETERS
# --------------------------------------------

# Path to the folder containing the 3D images
stimpath = r''

# Path to the folder where you want to save the images
outputpath = r''

# List all files in the directory
namesoffiles = listdir(stimpath)
timepoints = len(namesoffiles)

# Pixel size in microns (xy and z resolutions)
xyres = 2   # microns per pixel in x/y (Zeiss example)
zres = 2    # microns per pixel in z

# --------------------------------------------
# MAIN LOOP: PROCESS EACH IMAGE FILE
# --------------------------------------------

for t in range(timepoints):  # Loop through all experiments

    print('Segmenting experiment: ' + str(t))
    
    # Load the current 3D image (assumed TIFF stack)
    loadfile = join(stimpath, namesoffiles[t])
    resizedI = io.imread(loadfile) # shape: (z, y, x)
    #Our image now is a np.array
    
    # Define dimensions for convenience
    zdim, newy, newx = resizedI.shape
    
    #Start segmentation:
    
    # --------------------------------------------
    # 1. Estimate global threshold using multiple slices
    # --------------------------------------------
    # Select three representative slices in z, y, and x directions
    
    imageAz = resizedI[int(zdim/4),:,:]
    imageBz = resizedI[int(2*zdim/4),:,:]
    imageCz = resizedI[int(3*zdim/4),:,:]
    
    # Apply Gaussian blur to smooth images
    blurredAz = filters.gaussian(imageAz, sigma = 1)
    blurredBz = filters.gaussian(imageBz, sigma = 1)
    blurredCz = filters.gaussian(imageCz, sigma = 1)
    
    # Compute triangle threshold on blurred slices
    triangleAz = filters.threshold_triangle(blurredAz)
    triangleBz = filters.threshold_triangle(blurredBz)
    triangleCz = filters.threshold_triangle(blurredCz)
    
    # Repeat for y-slices
    
    imageAy = resizedI[:,int(newy/4),:]
    imageBy = resizedI[:,int(2*newy/4),:]
    imageCy = resizedI[:,int(3*newy/4),:]
    blurredAy = filters.gaussian(imageAy, sigma = 1)
    blurredBy = filters.gaussian(imageBy, sigma = 1)
    blurredCy = filters.gaussian(imageCy, sigma = 1)
    triangleAy = filters.threshold_triangle(blurredAy)
    triangleBy = filters.threshold_triangle(blurredBy)
    triangleCy = filters.threshold_triangle(blurredCy)
    
    # Repeat for x-slices
    
    imageAx = resizedI[:,:,int(newx/4)]
    imageBx = resizedI[:,:,int(2*newx/4)]
    imageCx = resizedI[:,:,int(3*newx/4)]
    blurredAx = filters.gaussian(imageAx, sigma = 1)
    blurredBx = filters.gaussian(imageBx, sigma = 1)
    blurredCx = filters.gaussian(imageCx, sigma = 1)
    triangleAx = filters.threshold_triangle(blurredAx)
    triangleBx = filters.threshold_triangle(blurredBx)
    triangleCx = filters.threshold_triangle(blurredCx)
    
    # Compute the average threshold across all slices
    
    meantriangle = np.mean([triangleAz, triangleBz, triangleCz, triangleAy, triangleBy, triangleCy, triangleAx, triangleBx, triangleCx])
    
    # --------------------------------------------
    # 2. Segment lumens by processing slices in Z, Y, and X directions
    # --------------------------------------------
    
    # Initialize empty arrays to store segmentations
    seg3dz = np.zeros_like(resizedI)
    vols3dz = np.zeros_like(resizedI)
    
    # ---- Z-axis segmentation ----
    
    for z in range(zdim):
        currentimage = resizedI[z,:,:]
        
        # Remove background using large Gaussian blur
        
        bground = filters.gaussian(currentimage, sigma = 20)
        enhanced = currentimage - bground
    
        # Smooth image
        blurred = filters.gaussian(currentimage, sigma = 1)
    
        # Enhance edges using Meijering filter (detects tubular structures)
        lines = filters.meijering(enhanced, sigmas = np.linspace(0,3,2), black_ridges = False) #enhanced lines
       
        #Threshold to lines image
        otsulines = filters.threshold_otsu(lines)
    
        # Apply global threshold and line segmentation
        
        triimage = blurred>meantriangle*1.2 #can be edited to get more accurate segmentation
        segmentedlines = lines>otsulines*0.8 #can be edited to get more accurate segmentation
        
        # Erode to remove outer organoid border
        erosionorg = ndi.binary_erosion(triimage, iterations = 20)
        
        # Combine lumens = lines + eroded organoid region
        lumens = segmentedlines*erosionorg
        
        # Fill holes inside lumens
        lumensfilled = ndi.binary_fill_holes(lumens)
        
        # Save result
        seg3dz[z,:,:] = lumensfilled
        vols3dz[z,:,:] = triimage
    print("Z done")
   
    seg3dy = np.zeros_like(resizedI)
    vols3dy = np.zeros_like(resizedI)
    
    # ---- Y-axis segmentation ----
    
    for y in range(newy):

        currentimage = resizedI[:,y,:]
        bground = filters.gaussian(currentimage, sigma = 20)
           
        #Removing background
        enhanced = currentimage - bground
    
        #Gaussian blur:
        blurred = filters.gaussian(currentimage, sigma = 1)
    
        #We use meijenring filter to enhance borders
        lines = filters.meijering(enhanced, sigmas = np.linspace(0,3,2), black_ridges = False) #enhanced lines
        #Threshold to lines image
        otsulines = filters.threshold_otsu(lines)
    
        #We have the threshold values but how do we get the image:    
        triimage = blurred>meantriangle*1.2 #can be edited to get more accurate segmentation
        segmentedlines = lines>otsulines*0.8 #can be edited to get more accurate segmentation
        # Let's get rid of the borders of the organoid:
            
        erosionorg = ndi.binary_erosion(triimage, iterations = 20)
        #We can get the final lumens by multiplying erodedbysegmented lines
        lumens = segmentedlines*erosionorg
        lumensfilled = ndi.binary_fill_holes(lumens)
        seg3dy[:,y,:] = lumensfilled
        vols3dy[:,y,:] = triimage
    #We want to save both images in one: z,chan,x,y
    print("Y done")
    seg3dx = np.zeros_like(resizedI)
    vols3dx = np.zeros_like(resizedI)
    
    # ---- X-axis segmentation ----
    
    for x in range(newx):
        currentimage = resizedI[:,:,x]
        bground = filters.gaussian(currentimage, sigma = 20)
           
        #Removing background
        enhanced = currentimage - bground
    
        #Gaussian blur:
        blurred = filters.gaussian(currentimage, sigma = 1)
    
        #We use meijenring filter to enhance borders
        lines = filters.meijering(enhanced, sigmas = np.linspace(0,3,2), black_ridges = False) #enhanced lines
        #Threshold to lines image
        otsulines = filters.threshold_otsu(lines)
    
        #We have the threshold values but how do we get the image:    
        triimage = blurred>meantriangle*1.2 #can be edited to get more accurate segmentation
        segmentedlines = lines>otsulines*0.8 #can be edited to get more accurate segmentation
        # Let's get rid of the borders of the organoid:
            
        erosionorg = ndi.binary_erosion(triimage, iterations = 20)
        #We can get the final lumens by multiplying erodedbysegmented lines
        lumens = segmentedlines*erosionorg
        lumensfilled = ndi.binary_fill_holes(lumens)
        seg3dx[:,:,x] = lumensfilled
        vols3dx[:,:,x] = triimage
    print("X done")
    
    # --------------------------------------------
    # 3. Combine all segmentation results
    # --------------------------------------------
    
    
    lumensum = seg3dz + seg3dy + seg3dx
    lumensum = lumensum.astype("uint8")
    
    # Final hole filling across all dimensions
    
    lumenfilt = lumensum > 1
    
    for z in range(zdim):
        lumenfilt[z,:,:] = ndi.binary_fill_holes(lumenfilt[z,:,:])
    for y in range(newy):
        lumenfilt[:,y,:] = ndi.binary_fill_holes(lumenfilt[:,y,:])
    for x in range(newx):
        lumenfilt[:,:,x] = ndi.binary_fill_holes(lumenfilt[:,:,x])
    
    lumenfilt = lumenfilt.astype("uint8")
    
    # Compute organoid shape as intersection of volume masks
    
    organoidshape = vols3dx*vols3dz*vols3dy
    
    # Prepare final multi-channel image: (z, channels, y, x)
    
    final = np.zeros((zdim,3, newy, newx), dtype = "uint8")
    
    # Normalize original image for visualization
    
    resizedI2 = (resizedI.astype(float) * 255 / np.max(resizedI)).astype('uint8')
    final[:,0,:,:] = resizedI2.astype("uint8")
    final[:,1,:,:] = lumenfilt.astype("uint8")
    final[:,2,:,:] = organoidshape.astype("uint8")
    
    # --------------------------------------------
    # 4. Save result as a multi-channel TIFF
    # --------------------------------------------
    
    imwrite(outputpath  + "/" + namesoffiles[t][:-4] + r"_3dseg.tif", final, imagej = True)