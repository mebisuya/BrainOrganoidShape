#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:44:32 2025

@author: marchenk
"""

#Code written by Marina Marchenko and Guillermo Martínez-Ara

#Short code to do 3D segmentation of organoids based on pretrained model

# Speed optimization libraries
from numba import jit, cuda

# Load model dependencies and libraries
import segmentation_models_pytorch as smp
from torch import nn
import torch
import torchvision
from torch.nn import functional as F
from tifffile import imwrite, TiffFile
from os import listdir, mkdir, remove
from os.path import join, exists
import matplotlib.pyplot as plt
import numpy as np

import os

# Allow multiple OpenMP libraries (prevents crash in some environments)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage import transform, filters, measure
import scipy.ndimage as ndi

# Check if CUDA (GPU) is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. Load pre-trained U-Net model
# -------------------------------

model = smp.Unet(
    encoder_name='resnet18',   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

# Load model weights

modelpath = r'\\ebisuya.embl.es\ebisuya\Marusja\3DLumen\Code\ModelVer09032023\ModelVer09032023\modlparam.zip' #folder containing the zip file modlparam.zip
model.load_state_dict(torch.load(modelpath,map_location=torch.device('cpu')))
model= model.to(DEVICE) # Move to GPU if available

    
# -------------------------------
# 2. Utility functions
# -------------------------------

def normT(array):
    #Normalization for Tensors
    valmax = torch.max(array)
    valmin = torch.min(array)
    normed = (array-valmin)/(valmax-valmin)
    return normed

def norm(array):
    #Normalization for NumPy arrays
    valmax = np.max(array)
    valmin = np.min(array)
    normed = ((array-valmin)/(valmax-valmin)).astype("float16")
    return normed

def Segment2D(image):
    #You must provide a 512x512 8bit normalized image, numpy format
    #We will first convert it to torch tensor
    x=torch.tensor(image).to(DEVICE)
    model.eval()
    while len(x.shape)<4:
        x = x.unsqueeze(0)
    # turn off gradient tracking
    with torch.no_grad():
        pred = model(x)
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    predim = pred.cpu().detach().numpy()
    
    return predim

def PercentileNorm(im, per = 99.95):
    #Input is a float array, output will be a normalized 8bit int array.
    
    newmax=np.percentile(im,per)
    im2=np.where(im<newmax,im/newmax,1)
    #Change it to 8 bit!
    im2 = (im2).astype("float32")
    return im2

def Segment3D(stack,threshold = 0.5):
    #To be fed with a 3D stack zx512x512 8bit not normalized image
    print("Segmenting Organoid")
    z,y,x= stack.shape
    
    tri=filters.threshold_triangle(stack)
    #Add pads
    maxdim = np.max([z,y,x])
    zpads = ((maxdim - z) // 2, (maxdim - z) // 2 + (maxdim - z) % 2)
    ypads = ((maxdim - y) // 2, (maxdim - y) // 2 + (maxdim - y) % 2)
    xpads = ((maxdim - x) // 2, (maxdim - x) // 2 + (maxdim - x) % 2)
    
    #We start by segmenting the image with triangle Threshold

    orgseg = stack>tri
    orgseg = ndi.binary_fill_holes(orgseg)
    label, nreg = ndi.label(orgseg)
    Props = measure.regionprops(label)
    areas = []
    for a in range(nreg):
        areas.append(Props[a].area)
    biggest = np.where(areas==np.amax(areas))[0][0]
    orgsegfin = np.where(label==biggest+1,1,0).astype("float32")
    
    proppix = np.sum(orgsegfin)/(z*x*y)
    print("Organoid takes the " +"{:.2f}".format(proppix*100) +"% of the stack")
    
    #Normalize
    stack = PercentileNorm(stack, per=100*(1-proppix*0.005))
    
    print("Segmenting Lumen")
    finalxy = np.zeros_like(stack, dtype="float16")
    for i in range(z):
        finalxy[i,:,:]=Segment2D(stack[i,:,:])
    #Normalize all data so it's float from 0 to 1.
    finalxy=np.nan_to_num(finalxy)
    finalxy = norm(finalxy)
    #Binarize to threshold
    finalxy = filters.gaussian(finalxy,sigma=2, preserve_range=True)
    finalxyb = (finalxy>threshold).astype("float16")
    print("Z done")
    
    finalyz = np.zeros_like(stack, dtype="float16")
    for i in range(y):
        finalyz[:,i,:]=Segment2D(stack[:,i,:])
    #Normalize all data so it's float from 0 to 1.
    finalyz=np.nan_to_num(finalyz)
    finalyz = norm(finalyz)
    #Binarize to threshold
    finalyz = filters.gaussian(finalyz,sigma=2, preserve_range=True)
    finalyzb = (finalyz>threshold).astype("float16")
    print("Y done")
    
    
    finalxz = np.zeros_like(stack, dtype="float16")
    for i in range(x):
        finalxz[:,:,i]=Segment2D(stack[:,:,i])
    #Normalize all data so it's float from 0 to 1.
    finalxz=np.nan_to_num(finalxz)
    finalxz = norm(finalxz)
    #Binarize to threshold
    finalxz = filters.gaussian(finalxz,sigma=2, preserve_range=True)
    finalxzb = (finalxz>threshold).astype("float16")
    print("X done")
    
    enhview= (finalxy+finalyz+finalxz)/3
    segview = ((finalxyb+finalyzb+finalxzb)>2).astype("float16")
    segview = orgsegfin*segview
    return stack, enhview, segview, orgsegfin

#Another layer of function, to perform the task on a 3D image, which is isometric

# -------------------------------
# 4. Process directory of TIFF stacks
# -------------------------------

filepath = r'C:\Path\To\Your\Data' # Fill this in
zres = 2.0 # Define your voxel size in microns
savepath = join(filepath,"results") #the output folder will be created automatically!

# Create/empty results folder
if not exists(savepath):
    mkdir(savepath)
oldfiles = listdir(savepath)
for f in oldfiles:
    remove(join(savepath,f))

filelist = listdir(filepath)
allfiles =[]
orgvolume = []
lumenvolume = []
nlumens = []
lumentoorgvol = []


# Loop over each TIFF file

for i in filelist:
    print("We are on file " + str(i))
    curpath = join(filepath,i)
    if ".tif" not in curpath:
        continue
    I1=TiffFile(curpath).asarray()
    nI = I1[:,:,:]
    z,y,x = nI.shape

    #Segment
    im,enh,seg,org = Segment3D(nI, 0.3)
    
    #Some quick measurements!
    orgvolume.append(np.sum(org)*zres*zres*zres)
    lumenvolume.append(np.sum(seg.astype("float32"))*zres*zres*zres)
    lumentoorgvol.append(np.sum(seg.astype("float32"))/np.sum(org))
    label, nlums = ndi.label(seg.astype("bool"))
    Props = measure.regionprops(label)
    #Let´s already add a filter for the lumens that are bigger than 100um3
    totlums = 0
    for a in range(nlums):
        if Props[a].area>100*zres*zres*zres:
            totlums +=1
    
    nlumens.append(totlums)
    
    
    savefile = join(savepath, "segmented"+i)
    fz,fy,fx = im.shape
    final = np.zeros((fz,4,fy,fx), dtype="float32")
    final[:,0,:,:] = im
    final[:,1,:,:] = enh
    final[:,2,:,:] = seg
    final[:,3,:,:] = org
    
    
    imwrite(savefile,final, imagej = True)
    allfiles.append(final)
    
#Finally transform the final one into a timelames

FINAL = np.stack(allfiles[:])
savefile = join(savepath, "segmented.tif")
imwrite(savefile,FINAL, imagej = True)