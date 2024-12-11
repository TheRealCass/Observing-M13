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


# pick a random day
random_number = random.randint(1,2)

if (random_number == 1): 

    random_file_index = random.randint(0,len(light1B_files)-1)
    ref_img_for_b_filter = fits.open(light1B_files[random_file_index])

else:
    
    random_file_index = random.randint(0,len(light2B_files)-1)
    ref_img_for_b_filter = fits.open(light2B_files[random_file_index])



if (random_number == 1):
 
    random_file_index = random.randint(0,len(light1R_files)-1)
    ref_img_for_r_filter = fits.open(light1R_files[random_file_index])

else:

    random_file_index = random.randint(0,len(light2R_files)-1)
    ref_img_for_r_filter = fits.open(light2R_files[random_file_index])


#get the header and the actual image from the choosen reference image
#get the header and the actual image from the choosen reference image
ref_header_b_filter   = ref_img_for_b_filter[0].header # this header will be stamped on any fits I make
ref_header_r_filter   = ref_img_for_r_filter[0].header # this header will be stamped on any fits I make

ref_data_b_filter     = ref_img_for_b_filter[0].data   # this is be use for allighining light frames
ref_data_r_filter     = ref_img_for_r_filter[0].data   # this is be use for allighining light frames

# make a list of all the B and R filter of the lights 
b_filter_lights = master_light_1B + master_light_2B
r_filter_lights = master_light_1R + master_light_2R

# alligh the lights using that random image we picked
master_Bfilter_light = align_image(b_filter_lights, ref_data_b_filter)
master_Rfilter_light = align_image(r_filter_lights, ref_data_r_filter)

# create the fits file to work with
b_filter_fits = fits.PrimaryHDU(master_Bfilter_light, ref_header_b_filter)
r_filter_fits = fits.PrimaryHDU(master_Rfilter_light, ref_header_r_filter)

# Find the peak locations in the blue and red filter images
peak_location_b_filter = find_peaks(b_filter_fits.data)
peak_location_r_filter = find_peaks(r_filter_fits.data)


# Define the World Coordinate System (WCS) from the blue filter image header
true_wcs_b_filter = WCS(b_filter_fits.header)
true_wcs_r_filter = WCS(r_filter_fits.header)

# Calculate the field of view (FoV) for the blue filter image
fov_b_filter = (b_filter_fits.data.shape * proj_plane_pixel_scales(true_wcs_b_filter))[0]
fov_r_filter = (r_filter_fits.data.shape * proj_plane_pixel_scales(true_wcs_r_filter))[0]


# Calculate statistics (mean, median, and standard deviation) for blue and red filter images
b_mean, b_median, b_std = sigma_clipped_stats(b_filter_fits.data, sigma = 3.0)
r_mean, r_median, r_std = sigma_clipped_stats(r_filter_fits.data, sigma = 3.0)

####### DAY 2 ########################

# Get RA and DEC coordinates from Gaia catalog within a radius
all_radecs_b_filter = gaia_radecs((b_filter_fits.header['RA'], b_filter_fits.header['DEC']), 1.2 * 0.65)
all_radecs_r_filter = gaia_radecs((r_filter_fits.header['RA'], r_filter_fits.header['DEC']), 1.2 * 0.65)

# we only keep stars 0.01 degree apart from each other
all_radecs_b_filter = sparsify(all_radecs_b_filter, 0.01)
all_radecs_r_filter = sparsify(all_radecs_r_filter, 0.01)


# Compute the WCS transformation using the peak locations and RA/DEC coordinates
wcs_b_filter = compute_wcs(peak_location_b_filter, all_radecs_b_filter[0:30], tolerance = 10)
wcs_r_filter = compute_wcs(peak_location_r_filter, all_radecs_r_filter[0:30], tolerance = 10)

# Update the blue & red filter image header with the new WCS information
b_filter_fits.header.update(wcs_b_filter.to_header())
r_filter_fits.header.update(wcs_r_filter.to_header())

# Normalize the blue & red filter image data
b_filter_fits.data = b_filter_fits.data / (16 * 120.0)
r_filter_fits.data = r_filter_fits.data / (16 * 120.0)

# locate the place you wanna save the fits files
output_directory_b_filter = DIRECTORY + "Light_B.fits"
output_directory_r_filter = DIRECTORY + "Light_R.fits"

# write the files to drive
b_filter_fits.writeto(output_directory_b_filter, overwrite=True)
b_filter_fits.writeto(output_directory_r_filter, overwrite=True)

# Get the co-orditane of M2 stars and their brightness from Database
# load the data from memory
coords = pd.read_csv("NGC7089.pos", sep='[ ]{2,}', engine='python')
mags = pd.read_csv("NGC7089.pho", sep='[ ]{1,}', engine='python')

# add coloum names to the coordinate file
coords.rename(columns={'323.36209232807' : 'RA', '-00.84555554920' : 'DEC'}, inplace=True)
coords.DEC = coords.DEC.str.replace(" ", "")
coords.DEC = coords.DEC.astype(float)
# Join the 2 tables using the "Reference" column as point of commonality
stetson = pd.merge( coords[["Reference", 'RA', 'DEC']], mags, how = 'inner', on = ['Reference'] )
# Keep the stars with a magnitude of 18 or brighter (mag < 19)
stetson = stetson[(stetson.B < 19) & (stetson.R < 19)]
# Lets see the detected stars from the steton photometric catalog
stetson_pos_b_filter = np.array(wcs_b_filter.world_to_pixel_values(stetson[["RA", "DEC"]]))
stetson_pos_r_filter = np.array(wcs_r_filter.world_to_pixel_values(stetson[["RA", "DEC"]]))

ax = plt.subplot(projection = wcs_b_filter)
ax.imshow(b_filter_fits.data, vmin = b_median, vmax = 3 * b_median, cmap = BLUE_FILTER_COLOR )
_ = CircularAperture(stetson_pos_b_filter, 20).plot(color="g", alpha=0.5)
ax = plt.subplot(projection = wcs_r_filter)
ax.imshow(r_filter_fits.data, vmin = b_median, vmax = 3 * b_median, cmap = RED_FILTER_COLOR )
_ = CircularAperture(stetson_pos_b_filter, 20).plot(color = "g", alpha = 0.5)
