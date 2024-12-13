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
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import angular_separation, Angle, SkyCoord
from astropy.stats import sigma_clipped_stats
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
#Constnats
# BLUE_FILTER_COLOR =  'Blues'
# RED_FILTER_COLOR =  'Reds'

BLUE_FILTER_COLOR =  cm.grey
RED_FILTER_COLOR =  cm.grey
# Target
DIRECTORY="m2/"
# DIRECTORY="m13/"
# Gather all the files
#Get the file directory

# Define the directories where the FITS files are stored
directory_b_filter = DIRECTORY + "Light_B.fits"

# Load the FITS files into variables
b_filter_fits = fits.open(directory_b_filter)
b_filter_fits = b_filter_fits[0]

# Use twirl to find peaks
peak_location_b_filter = find_peaks(b_filter_fits.data)[0:20]

# Obtain and filter RA/DEC coordinates from the Gaia catalog
b_filter_center     = (b_filter_fits.header['RA'], b_filter_fits.header['DEC']) 
fov_b_filter        = 1.2 * 0.65
all_radecs_b_filter = gaia_radecs(b_filter_center, fov_b_filter)
thrushold           = 0.01 # degrees
all_radecs_b_filter = sparsify(all_radecs_b_filter, thrushold)

# Compute WCS coordinates
# we only keep the 12 brightest stars from gaia
wcs_b_filter = compute_wcs(peak_location_b_filter, all_radecs_b_filter[0:30], tolerance = 10)

# Update Header for the lights with the new WCS
b_filter_fits.header.update(wcs_b_filter.to_header())
# Normalize the lights
b_filter_fits.data = b_filter_fits.data/(16*120.0)
# Get the co-orditane of M2 stars and their brightness from Database (CANFAR's Stetson Database)
# load the data from memory
coords = pd.read_csv( DIRECTORY + "NGC7089.pos", sep='[ ]{2,}', engine='python') # replace anything w 2 or more spaces with just a space
mags = pd.read_csv(DIRECTORY + "NGC7089.pho", sep='[ ]{1,}', engine='python') # replace anything w 1 or more spaces with just a space
# Let's extract what we need from the stetson database
# add coloum names to the coordinate file
coords.rename(columns={'323.36209232807' : 'RA', '-00.84555554920' : 'DEC'}, inplace=True) # replace the orgonal col names for somethig meaningfull

# Convert the DEC column to string type first
coords['DEC'] = coords['DEC'].astype(str)

# Now you can use the .str accessor to replace spaces
coords['DEC'] = coords['DEC'].str.replace(" ", "")

# Convert the DEC column back to float type
coords['DEC'] = coords['DEC'].astype(float)

# Perform the merge with the photometry data
stetson = pd.merge(coords[["Reference", 'RA', 'DEC']], mags[["Reference", 'B', 'sigma.1', 'R', 'sigma.3']], how='inner', on=['Reference'])

# Filter the stars with magnitude of 18 or brighter
stetson = stetson[(stetson.B < 19) & (stetson.R < 19)]

# Lets plot what we extracted
stetson_pos_b_filter = np.array(wcs_b_filter.world_to_pixel_values( stetson[["RA", "DEC"]] ))
ax1 = plt.subplot(projection = wcs_b_filter)
ax1.imshow(b_filter_fits.data, vmin = np.median(b_filter_fits.data), vmax = 3 * np.median(b_filter_fits.data), cmap = BLUE_FILTER_COLOR)
_ = CircularAperture(stetson_pos_b_filter, 10).plot(color = "g", alpha = 0.5)
ax1.set_title('Blue Filter')

plt.show()
plt.savefig(DIRECTORY + "Steton Catalog Stars.jpg")
plt.clf()
# Lets define & Work with a small subsection
#### FOV too large
position = (2200, 1000)
size = (1000, 1000)
cutout_b_filter = Cutout2D(b_filter_fits.data, position, size, wcs = wcs_b_filter)

b_filter_analysis = fits.PrimaryHDU()
b_filter_analysis.data = cutout_b_filter.data
b_filter_analysis.header.update(cutout_b_filter.wcs.to_header())

# Use TWIRL to compute WCS for the cutout
#compute wcs for this cutout
peaks_b_filter = find_peaks(b_filter_analysis.data)[0:20]
wcs_cutout_b_filter = compute_wcs(peaks_b_filter, all_radecs_b_filter[0:30], tolerance = 10)
stetson_pos_b_filter = np.array(wcs_cutout_b_filter.world_to_pixel_values(stetson[["RA", "DEC"]]))

# Find peaks using DAO Star Finder
mean, median, std = sigma_clipped_stats(b_filter_analysis.data, sigma = 3.0)

daofind = DAOStarFinder(fwhm = 7.0, threshold = 5.*std)
sources = daofind(b_filter_analysis.data - median)
for col in sources.colnames:  
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # for consistent table output
# Lets convert the DAOStarFinder results into a pandas table & get what we need
detected_stars      = sources.to_pandas()
detected_stars      = detected_stars[(detected_stars.flux > 0.79)] # Get rid of anything fainter than 0.79 mag
detected_locaions   = np.transpose((detected_stars['xcentroid'], detected_stars['ycentroid'])) # pandas object backt o numpy array

#  Overlay the Stetson star locations & DAO Star Finder peak locations on the cutout

plt.imshow(b_filter_analysis.data, vmin = np.median(b_filter_fits.data), vmax = 3 * np.median(b_filter_fits.data), cmap= "Greys_r")
_ = CircularAperture(stetson_pos_b_filter, 10).plot(color="g", alpha=0.5)
_ = CircularAperture(detected_locaions, 10).plot(color="m", alpha=0.5)
ax1.set_title('Peaks found using TWIRL in B filter cutout')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
plt.savefig(DIRECTORY + "TWIRL peaks in curout.jpg")
plt.clf()
# Add the detected coordinates from TWIRL into the to the DAOStarFinder results
#get RA and DEC in degrees (SkyCoord Object) in the detected steson stars
detected_stars['coords'] = wcs_cutout_b_filter.pixel_to_world(detected_stars['xcentroid'], detected_stars['ycentroid'])
coordinates_stetson = SkyCoord(stetson.RA, stetson.DEC, unit=u.degree) 

#get RA and DEC in degrees (SkyCoord Object) in the DAOStarFinder peaks
det_radec = wcs_cutout_b_filter.pixel_to_world(detected_stars['xcentroid'], detected_stars['ycentroid'])
coordinates_detected = SkyCoord(det_radec.ra, det_radec.dec)
# Match with stars that are present in both tables based on SkyCoordinate Object
idx, d2d, d3d = coordinates_detected.match_to_catalog_sky(coordinates_stetson)
stetson['sky'] = SkyCoord(stetson.RA, stetson.DEC, unit=u.degree)

max_sep = 1.0 * u.arcsec
sep_constraint = d2d < max_sep
coordinates_matched = coordinates_detected[sep_constraint]
catalog_matches = coordinates_stetson[idx[sep_constraint]]
detected_stars['coords'] = coordinates_detected

# Get X axis
detected_and_matched_stars_mag = []
for i in detected_stars.index:
    for j in range(0,len(coordinates_matched)):
        if detected_stars.coords[i] == coordinates_matched[j]:
            detected_and_matched_stars_mag.append(detected_stars.daofind_mag[i])

stetson_and_matched_stars_r_band = []
for i in stetson.index:
    for j in range(0,len(catalog_matches)):
        if(stetson.sky[i] == catalog_matches[j]):
            stetson_and_matched_stars_r_band.append(stetson.R[i])

# get Y axix
apparent_mag = []
for i in stetson.index:
    for j in range(0,len(catalog_matches)):
        if(stetson.sky[i] == catalog_matches[j]):
            apparent_mag.append(stetson.B[i])

# Assuming stetson_and_matched_stars_r_band and detected_and_matched_stars_mag are lists
stetson_and_matched_stars_r_band = np.array(stetson_and_matched_stars_r_band)
detected_and_matched_stars_mag = np.array(detected_and_matched_stars_mag)

# Now you can subtract these arrays
difference = stetson_and_matched_stars_r_band - detected_and_matched_stars_mag

# Assuming apparent_mag is already defined and is not a list
plt.scatter(difference, apparent_mag)
plt.xlabel('Difference (Stetson - Detected)')
plt.ylabel('Apparent Magnitude')
plt.title('Star Magnitude Comparison')
plt.show()