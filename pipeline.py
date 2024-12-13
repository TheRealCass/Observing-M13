# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import astropy.units as u
from astropy.visualization import simple_norm, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import angular_separation, Angle, SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.nddata import Cutout2D
import os
import astroalign as align
from twirl import find_peaks
from twirl import gaia_radecs
from twirl.geometry import sparsify
from twirl import compute_wcs
from photutils.aperture import SkyCircularAperture
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from pylab import figure, cm
import pandas as pd
import random

#Constnats
# BLUE_FILTER_COLOR =  'Blues'
# RED_FILTER_COLOR =  'Reds'
BLUE_FILTER_COLOR =  cm.grey
RED_FILTER_COLOR =  cm.grey

# Target
DIRECTORY="m2/"
# DIRECTORY="m13/"

# Gather all the files
darks_directory     = DIRECTORY + "darks/120/"
bias_directory      = DIRECTORY + "bias/2x2/"
flats1B_directory   = DIRECTORY + "Night 1/flats/B"
flats1R_directory   = DIRECTORY + "Night 1/flats/R"
flats2B_directory   = DIRECTORY + "Night 2/flats/B"
flats2R_directory   = DIRECTORY + "Night 2/flats/R"
light1B_directory   = DIRECTORY + "Night 1/B"
light1R_directory   = DIRECTORY + "Night 1/R"
light2B_directory   = DIRECTORY + "Night 2/B"
light2R_directory   = DIRECTORY + "Night 2/R"

#Create arays of file names, it's easier to reference them
darks_files         = get_filename(darks_directory)
bias_files          = get_filename(bias_directory)

flats1B_files       = get_filename(flats1B_directory)
flats1R_files       = get_filename(flats1R_directory)
flats2B_files       = get_filename(flats2B_directory)
flats2R_files       = get_filename(flats2R_directory)

light1B_files       = get_filename(light1B_directory)
light1R_files       = get_filename(light1R_directory)
light2B_files       = get_filename(light2B_directory)
light2R_files       = get_filename(light2R_directory)

# Make master bias and darks
master_bias = stack_frames(bias_files) 
master_dark = stack_frames(darks_files)

# Make master flats for each night on each filter
flat1B_img = stack_frames(flats1B_files)
flat1R_img = stack_frames(flats1R_files)
flat2B_img = stack_frames(flats2B_files)
flat2R_img = stack_frames(flats2R_files)

#Subtract the bias from the flats
flat1B_bias = flat1B_img - master_bias 
flat1R_bias = flat1R_img - master_bias
flat2B_bias = flat2B_img - master_bias
flat2R_bias = flat2R_img - master_bias

#Normalize by dividing by the maximum value of the array
master_flat_1B = flat1B_bias / np.max(flat1B_bias) 
master_flat_1R = flat1R_bias / np.max(flat1R_bias)
master_flat_2B = flat2B_bias / np.max(flat2B_bias)
master_flat_2R = flat2R_bias / np.max(flat2R_bias)

# Correct Lights
master_light_1B = correct_lights(light1B_files, master_dark, master_flat_1B)
master_light_1R = correct_lights(light1R_files, master_dark, master_flat_1R)
master_light_2B = correct_lights(light2B_files, master_dark, master_flat_2B)
master_light_2R = correct_lights(light2R_files, master_dark, master_flat_2R)

# Choose a reference image
ref_img_for_b_filter = fits.open(light1B_files[0])
ref_img_for_r_filter = fits.open(light1R_files[0])

# Get the data off the reference image
ref_header_b_filter   = ref_img_for_b_filter[0].header 
ref_header_r_filter   = ref_img_for_r_filter[0].header 
ref_data_b_filter     = ref_img_for_b_filter[0].data  
ref_data_r_filter     = ref_img_for_r_filter[0].data


# Alighn light frame from each filters
b_filter_lights = master_light_1B + master_light_2B
r_filter_lights = master_light_1R + master_light_2R
master_Bfilter_light = align_image(b_filter_lights, ref_data_b_filter)
master_Rfilter_light = align_image(r_filter_lights, ref_data_r_filter)

# create the fits file to work with
b_filter_fits = fits.PrimaryHDU(master_Bfilter_light, ref_header_b_filter)
r_filter_fits = fits.PrimaryHDU(master_Rfilter_light, ref_header_r_filter)

# Use twirl to find peaks
peak_location_b_filter = find_peaks(b_filter_fits.data)[0:20]
peak_location_r_filter = find_peaks(r_filter_fits.data)[0:20]

# Obtain and filter RA/DEC coordinates from the Gaia catalog
all_radecs_b_filter = gaia_radecs((b_filter_fits.header['RA'], b_filter_fits.header['DEC']), 1.2 * 0.65)
all_radecs_r_filter = gaia_radecs((r_filter_fits.header['RA'], r_filter_fits.header['DEC']), 1.2 * 0.65)

# we only keep stars 0.01 degree apart from each other
thrushold           = 0.01 # degrees
all_radecs_b_filter = sparsify(all_radecs_b_filter, thrushold)
all_radecs_r_filter = sparsify(all_radecs_r_filter, thrushold)

# Map WCS coordinates to image coordinate
num_of_stars_to_keep = 30
wcs_b_filter = compute_wcs(peak_location_b_filter, all_radecs_b_filter[0:num_of_stars_to_keep], tolerance = 10)
wcs_r_filter = compute_wcs(peak_location_r_filter, all_radecs_r_filter[0:num_of_stars_to_keep], tolerance = 10)

# Update Header for the lights with the new WCS
b_filter_fits.header.update(wcs_b_filter.to_header())
r_filter_fits.header.update(wcs_r_filter.to_header())

# Normalize the lights
b_filter_fits.data = b_filter_fits.data/(16*120.0)
r_filter_fits.data = r_filter_fits.data/(16*120.0)

# Save the Lights as FITS files
output_directory_b_filter = DIRECTORY + "Light_B.fits"
output_directory_r_filter = DIRECTORY + "Light_R.fits"
b_filter_fits.writeto(output_directory_b_filter, overwrite = True)
r_filter_fits.writeto(output_directory_r_filter, overwrite = True)

# LAB 02
