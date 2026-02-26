# Lumen-Organoid-Segmentation-Measurements

### Introduction
This repository contains the full pipeline for making images isometric, image segmentation and measuremens of the parameters of Lumens and Organoids

### Pipeline
1. After you finished imaging your samples you will have images with full resolution. First, you need to prepare your image for the segmentation
   - stitch all the tiles
   - fuse multiangle image (by using Angle-fusion-tif code)
   - resize the images to the width 512 pixels
   - find out the Properties of your final images: z, y, x resolution (Can be done in the Fiji - Image - Properties tab)
     
2. Proceed to the image processing step
   
4. Open code "ToMakeImageIsotropic-updated". This code will make equal resolution in x, y, z. We need it to simplify our workflow and further measurements. Please read the comments inside the code to use it.

Alternatively, you can use Fiji Image-Scale function, and enter Z dimention manually (to calculate new Z dimention you need to multiply Z stack number (like 643) by zresolution and divide by x or y resolution, they are the same)
   
6. Now images are ready to be segmented. You have two options for the segmentation.
   - First is to use the code "LumenAndOrganoidSegmentation-ver1". This code computes a global threshold by analyzing multiple slices in all 3 axes and segments lumens based on intensity and structure of the image. The output image has the following stracture:

Channel 0 = Original image

Channel 1 = Lumen mask

Channel 2 = Organoid volume

- Alternatively, you can use more powerful approach for the segmentation by using the code "LumenSegmentation-basedonAI". To use it you need to download the file ModelVer09032023 with modlparam.zip. This is a U-net model (a machine learning model) that was pre-trained to segment lumens.
- 
I trained it on my images if you want to customise it you should use "ModelTraining-Copy-1".

The output image has the following stracture:

Channel 0 = Original image

Channel 1 = Lumen mask that machine learning algorithm created

Channel 2 = Lumen mask

Channel 3 = Organoid mask

In general, second option worked better on my images.

7. Final step - Measurement of the parameters
Open the code "Newmesh-measurements". To measure the parameters you need to give it the images that contain only one channel with the mask (either lumens or organoid). It will measure the volume, surface area, integral mean curvature, sphericity and number of lumens or organoids in your images.
You will have several tables saved: all the lumens, the number of columns in the table equals the number of lumens in the organoid
