# SEM Microstructure Analyzer: Automated Particle Size Distribution Toolkit

## DOI
[![DOI](https://zenodo.org/badge/1167568433.svg)](https://doi.org/10.5281/zenodo.18975583)

## Overview
This repository contains a specialized computational framework for the automated morphological analysis of Scanning Electron Microscopy (SEM) images. Developed to support the study of silver-coated cenospheres, this tool provides an unbiased, reproducible pipeline for particle segmentation, size quantification, and statistical evaluation in microwave composite research.

The app is built using the **Streamlit** framework, providing an intuitive GUI for complex image processing routines involving OpenCV and SciPy.

## Key Features
- **Multi-Method Segmentation:** Adaptive Thresholding, Top-Hat transforms, and Otsu’s binarization for robust feature extraction.
- **Statistical Rigor:** Calculation of number-weighted, area-weighted, and volume-weighted distributions.
- **Advanced KDE Fitting:** Log-space Kernel Density Estimation (KDE) with bin-width scaling for mathematically consistent trend visualization.
- **Batch Processing:** Data pooling from multiple images to ensure representative statistical sampling.
- **High-Quality Export:** Direct export of processed data to CSV and publication-ready PDF reports.

## Methodology
The core logic utilizes an isotropic growth model for 3D volume reconstruction from 2D SEM projections. The statistical engine employs weighted KDE to align continuous probability trends with discrete fractional histograms on logarithmic axes, ensuring strict normalization ($\sum F_i = 1$).

## Installation
To run the analyzer locally, ensure you have Python 3.9+ installed, then:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sem-analyzer.git
   cd sem-analyzer
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
3. Launch the app:
    ```bash
    streamlit run app.py

## Requirements
streamlit
opencv-python-headless
numpy
pandas
matplotlib
scipy

## License
This project is licensed under the MIT License - see the LICENSE file for details.

