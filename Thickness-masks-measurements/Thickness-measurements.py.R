#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:51:30 2026

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
        # Area & perimeter
        # -----------------------------
        area = region.area
        perimeter = region.perimeter_crofton

        circularity = (
            4 * math.pi * area / (perimeter ** 2)
            if perimeter > 0 else np.nan
        )

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
                lumen_perimeter = lumen.perimeter_crofton

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

    for file_name in os.listdir(input_folder):

        if not file_name.lower().endswith((".tif", ".tiff")):
            continue

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
    
    
input_folder = r""
output_folder = r""

df = process_folder(
    input_folder=input_folder,
    output_folder=output_folder,
    output_csv_name="thickness_results.csv",
    #min_object_size=200
)

print(df.head())
    
    