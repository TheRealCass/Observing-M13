# Imports
# Import necessary libraries for mathematical operations and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

# Import astropy modules for working with astronomical data
import astropy.units as u
from astropy.visualization import simple_norm, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
from astropy.coordinates import angular_separation, Angle, SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
# Import os for directory operations
import os
# Import astroalign for aligning astronomical images
import astroalign as align
# Import twirl for peak finding and WCS (World Coordinate System) computations
from twirl import find_peaks
from twirl import gaia_radecs
from twirl.geometry import sparsify
from twirl import compute_wcs
# Import photutils for aperture photometry
from photutils.aperture import SkyCircularAperture
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from pylab import figure, cm
# Import pandas for data manipulation
import pandas as pd

import random
import helper

#Constnats
# BLUE_FILTER_COLOR =  'Blues'
# RED_FILTER_COLOR =  'Reds'

BLUE_FILTER_COLOR =  cm.grey
RED_FILTER_COLOR =  cm.grey
DIRECTORY="m2/" # Target
# DIRECTORY="m13/"


# Gather all the files
#Get the file directory
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
darks_files         = helper.get_filename(darks_directory)
bias_files          = helper.get_filename(bias_directory)

flats1B_files       = helper.get_filename(flats1B_directory)
flats1R_files       = helper.get_filename(flats1R_directory)
flats2B_files       = helper.get_filename(flats2B_directory)
flats2R_files       = helper.get_filename(flats2R_directory)

light1B_files       = helper.get_filename(light1B_directory)
light1R_files       = helper.get_filename(light1R_directory)
light2B_files       = helper.get_filename(light2B_directory)
light2R_files       = helper.get_filename(light2R_directory)
# Make master bias and darks
master_bias = helper.stack_frames(bias_files) 
master_dark = helper.stack_frames(darks_files)
# Make master flats
#stack flat images for each day & filter to creat 4 master flats
flat1B_img = helper.stack_frames(flats1B_files)
flat1R_img = helper.stack_frames(flats1R_files)
flat2B_img = helper.stack_frames(flats2B_files)
flat2R_img = helper.stack_frames(flats2R_files)

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
# Make master light
master_light_1B = helper.correct_lights(light1B_files, master_dark, master_flat_1B)
master_light_1R = helper.correct_lights(light1R_files, master_dark, master_flat_1R)
master_light_2B = helper.correct_lights(light2B_files, master_dark, master_flat_2B)
master_light_2R = helper.correct_lights(light2R_files, master_dark, master_flat_2R)
# Choose a reference image
#get a random image from all the light images to use as reference image
random_number = random.randint(1,4)

if (random_number == 0): 

    random_file_index = random.randint(0,len(light1B_files)-1)
    ref_img = fits.open(light1B_files[random_file_index])

elif (random_number == 2):
    
    random_file_index = random.randint(0,len(light1R_files)-1)
    ref_img = fits.open(light1R_files[random_file_index])

elif (random_number == 3):
 
    random_file_index = random.randint(0,len(light2B_files)-1)
    ref_img = fits.open(light1B_files[random_file_index])

else:

    random_file_index = random.randint(0,len(light2R_files)-1)
    ref_img = fits.open(light1R_files[random_file_index])

# Get the data off the reference image
#get the header and the actual image from the choosen reference image
ref_header   = ref_img[0].header # this header will be stamped on any fits I make
ref_data     = ref_img[0].data   # this is be use for allighining light frames
# Alighn light frame from each filters
# make a list of all the B and R filter of the lights 
b_filter_lights = master_light_1B + master_light_2B
r_filter_lights = master_light_1R + master_light_2R

# alligh the lights using that random image we picked
master_Bfilter_light = helper.align_image(b_filter_lights, ref_data)
master_Rfilter_light = helper.align_image(r_filter_lights, ref_data)

# create the fits file to work with
b_filter_fits = fits.PrimaryHDU(master_Bfilter_light, ref_header)
r_filter_fits = fits.PrimaryHDU(master_Rfilter_light, ref_header)


# Let's define a small subsection
# Define the crop dimensions
crop_x_start = 2300
crop_x_end = 2700
crop_y_start = 1000
crop_y_end = 1400

# Calculate the width and height based on crop dimensions
crop_width = crop_x_end - crop_x_start
crop_height = crop_y_end - crop_y_start


# Cut out the small subsection
# Crop the images based on the defined dimensions
cropped_b_filter_img = b_filter_fits.data[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
cropped_r_filter_img = r_filter_fits.data[crop_y_start:crop_y_end, crop_x_start:crop_x_end]


# Find the stats of the cropped images
# Calculate statistics (mean, median, and standard deviation) for the subsets of blue and red filter data
_, b_median, b_std = sigma_clipped_stats(cropped_b_filter_img, sigma = 3.0)
_, r_median, r_std = sigma_clipped_stats(cropped_r_filter_img, sigma = 3.0)

# Use stats in DAOStarFinder to locate sources
# initialise DAO Star Finder. Use the stats to find a suatitable thrushold
finder_for_b_filter = DAOStarFinder(fwhm = 5.0, threshold = 3.*b_std)
finder_for_r_filter = DAOStarFinder(fwhm = 5.0, threshold = 3.*r_std)

# make a background-subtracted image.
no_background_b_filter_img = cropped_b_filter_img - b_median
no_background_r_filter_img = cropped_r_filter_img - r_median

# run DAO star finder to find sources in B & R filter images that have been cropped
# DAOStarFinder returns a table with properties it found of each star it detected
b_filter_sources = finder_for_b_filter(no_background_b_filter_img)
r_filter_sources = finder_for_r_filter(no_background_r_filter_img)


# Let's take a look at what we found
# get the name of the colums that store the x and y position of the detected stars

#print the table to find the col names of interst








