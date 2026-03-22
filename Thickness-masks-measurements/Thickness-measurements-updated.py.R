#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:00:42 2026

@author: marchenk
"""

import math
import numpy as np
import tifffile as tif
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import os
import pandas as pd
from skimage.measure import regionprops_table

def analyze_thickness_tiff(
    tiff_path,
    min_object_size=50,
    connectivity=2
):
    img = tif.imread(tiff_path).astype(float)

    mask = img > 0
    mask = remove_small_objects(mask, min_size=min_object_size)

    labeled = label(mask, connectivity=connectivity)

    measurements = []

    for region in regionprops(labeled, intensity_image=img):

        # -----------------------------
        # Thickness statistics
        # -----------------------------
        thickness = region.intensity_image[region.intensity_image > 0]
        if thickness.size == 0:
            continue

        mean_thickness = thickness.mean()
        std_thickness = thickness.std()
        cv_thickness = std_thickness / mean_thickness if mean_thickness > 0 else np.nan

        # -----------------------------
        # Area, perimeter & Roundness (FIJI MATCHING)
        # -----------------------------
        # 1. Area: The actual pixels of the ring (as you defined in Fiji)
        area = region.area
        
        # 2. Perimeter: Use standard perimeter (better Fiji match than Crofton)
        perimeter = region.perimeter_crofton
        
        # 3. Major Axis: To match Fiji, we find the axis of the FILLED shape
        # region.image is the ring; region.filled_image is the ring + lumen
        filled_label = label(region.filled_image.astype(int))
        filled_props = regionprops(filled_label)
        
        if filled_props:
            # We take the major axis of the solid version of this object
            fiji_major_axis = filled_props[0].major_axis_length
        else:
            fiji_major_axis = region.major_axis_length
        
        # Fiji Roundness: 4 * Area / (pi * (MajorAxis of filled shape)^2)
        circularity = (4 * area) / (np.pi * (fiji_major_axis**2)) if fiji_major_axis > 0 else np.nan

        # -----------------------------
        # Lumen (hole) perimeter
        # -----------------------------
        object_mask = region.image  # binary mask in bounding box

        filled = binary_fill_holes(object_mask)
        holes = filled & (~object_mask)

        lumen_perimeter = np.nan

        if holes.any():
            hole_labels = label(holes, connectivity=connectivity)
            hole_regions = regionprops(hole_labels)

            if hole_regions:
                # Select largest hole = lumen
                lumen = max(hole_regions, key=lambda r: r.area)
                # Using standard perimeter for lumen too for consistency
                lumen_perimeter = lumen.perimeter

        measurements.append({
            "mean_thickness": mean_thickness,
            "std_thickness": std_thickness,
            "cv_thickness": cv_thickness,
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "circularity": circularity,
            "lumen_perimeter": lumen_perimeter
        })

    return measurements
    
def process_folder(
    input_folder,
    output_folder,
    output_csv_name="thickness_measurements.csv",
    min_object_size=50,
    connectivity=2
):
    all_results = []

    # Filter for tiff files
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    
    if not files:
        print(f"No TIFF files found in {input_folder}")
        return pd.DataFrame()

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)

        object_measurements = analyze_thickness_tiff(
            file_path,
            min_object_size=min_object_size,
            connectivity=connectivity
        )

        for idx, obj in enumerate(object_measurements, start=1):
            all_results.append({
                "file_name": file_name,
                "object_id": idx,
                **obj
            })

    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(output_folder, output_csv_name)
    df.to_csv(output_path, index=False)

    return df
    
# --- Execution ---
input_folder = r""  # Set your input path
output_folder = r"" # Set your output path

df = process_folder(
    input_folder=input_folder,
    output_folder=output_folder,
    output_csv_name="thickness_results.csv",
)

if not df.empty:
    print(df.head())