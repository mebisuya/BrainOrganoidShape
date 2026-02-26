#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:17:54 2025

@author: marchenk
"""

#Code written by Marina Marchenko

import numpy as np
from skimage import io, transform
from os import listdir, makedirs
from os.path import join, exists

# --- Input Parameters ---
# Path to the folder containing the 3D images
filepath = r''

# Path to the folder where you want to save the isotropic images
outputpath = r''
if not exists(outputpath):
    makedirs(outputpath)

# List all image files in the input directory
namesoffiles = [f for f in listdir(filepath) if f.lower().endswith(('.tif', '.tiff'))]

# Original voxel resolution (µm per pixel)
res_x, res_y, res_z = 0.4126, 0.4126, 2.0  # Example values

# Determine smallest resolution (target for isotropy)
target_res = min(res_x, res_y, res_z)
print(f"Target isotropic resolution: {target_res:.4f} µm/voxel")

# --- Process Each Image ---
for n, fname in enumerate(namesoffiles, 1):
    print(f"\n[{n}/{len(namesoffiles)}] Processing: {fname}")

    # Load the 3D image
    loadfile = join(filepath, fname)
    img = io.imread(loadfile)

    # Ensure the image is 3D (z, y, x)
    if img.ndim == 4:  # (z, channel, y, x)
        img = img[:, 0, :, :]  # Take the first channel if multiple exist
    elif img.ndim != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    zdim, ydim, xdim = img.shape

    # --- Compute new shape ---
    scale_z = res_z / target_res
    scale_y = res_y / target_res
    scale_x = res_x / target_res

    new_shape = (
        int(zdim * scale_z),
        int(ydim * scale_y),
        int(xdim * scale_x)
    )

    print(f"Original shape: {img.shape}")
    print(f"New isotropic shape: {new_shape}")

    # --- Resample image ---
    img_iso = transform.resize(
        img,
        new_shape,
        preserve_range=True,
        order=1,  # linear interpolation (use 0 for nearest, 3 for cubic)
        anti_aliasing=True
    )

    # --- Save isotropic image ---
    output_file = join(outputpath, f"isotropic_{fname}")
    io.imsave(output_file, img_iso.astype(img.dtype))
    print(f"✅ Saved isotropic image to: {output_file}")