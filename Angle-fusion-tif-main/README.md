# Angle-fusion-tif

### 🖼️ **Overview**

This repository provides a workflow to fuse two fluorescence images of the same sample with multiple channels, acquired from different directions. These images are usually misaligned, rotated, and exhibit decreasing signal along one axis. The pipeline corrects these issues and generates a single fused image.


## ✨ Features
- Align and fuse two fluorescence images acquired from different directions.
- Handles rotation and signal attenuation.
- Supports **multi-channel TIFF images**.
- Runs in **Jupyter Notebook** for interactive processing.

---

## 🛠 Installation

1. Install [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda).
  
2. Clone this repository:

git clone https://github.com/<your-username>/Flipped_Image_Fusion_tif.git
cd Flipped_Image_Fusion_tif

3. Create a Conda environment and install dependencies

mamba env create -f env.yml

4. Activate the environment

mamba activate elastix-napari

5. Then run the jupyter notebook with your favorite IDE (like VS Code, Pycharm, ...) or via Jupyter Lab by typing in the terminal:

`jupyter lab`

### 3. **Usage**

1. Open the Jupyter Notebook provided in the repository.

2. Update folder paths in the notebook:

Input folder: where your input TIFF images are located.

Output folder: where fused images will be saved.

3. Image naming convention:

Each pair should follow this format: sampleName-positionX-0degree.tif and sampleName-positionX-180degree.tif

Example:
antibody2-wt-position1-0degree.tif
antibody2-wt-position1-180degree.tif

4. Run the notebook cells to align and fuse the images.

5. File Format

Input: Multi-channel TIFF (.tif) images.
Output: Fused TIFF image.
