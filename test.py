# Import necessary libraries for mathematical operations and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

# Import helper module containing custom functions
import helper

# Define the directory containing the images
DIRECTORY = "m2/"

# Define paths to different sets of images within the directory
flats1B_directory = DIRECTORY + "Night 1/flats/B"
flats1R_directory = DIRECTORY + "Night 1/flats/R"
flats2B_directory = DIRECTORY + "Night 2/flats/B"
flats2R_directory = DIRECTORY + "Night 2/flats/R"
darks_directory = DIRECTORY + "darks/120/"   # Dark images
bias_directory = DIRECTORY + "bias/2x2/"     # Bias images
light1B_directory = DIRECTORY + "Night 1/B"
light1R_directory = DIRECTORY + "Night 1/R"
light2B_directory = DIRECTORY + "Night 2/B"
light2R_directory = DIRECTORY + "Night 2/R"

# Get all filenames in the specified directories using the helper function
flats1B_files = helper.get_filename(flats1B_directory)
flats1R_files = helper.get_filename(flats1R_directory)
flats2B_files = helper.get_filename(flats2B_directory)
flats2R_files = helper.get_filename(flats2R_directory)
darks_files = helper.get_filename(darks_directory)
bias_files = helper.get_filename(bias_directory)
light1B_files = helper.get_filename(light1B_directory)
light1R_files = helper.get_filename(light1R_directory)
light2B_files = helper.get_filename(light2B_directory)
light2R_files = helper.get_filename(light2R_directory)

# Create master images by averaging the respective sets of images
flat1B_img = helper.make_master(flats1B_files)
flat1R_img = helper.make_master(flats1R_files)
flat2B_img = helper.make_master(flats2B_files)
flat2R_img = helper.make_master(flats2R_files)
bias_img = helper.make_master(bias_files)
darks_img = helper.make_master(darks_files)

# Correct the flat images by subtracting the bias image
flat1B_bias = flat1B_img - bias_img
flat1R_bias = flat1R_img - bias_img
flat2B_bias = flat2B_img - bias_img
flat2R_bias = flat2R_img - bias_img

# Normalize the flat images by dividing by the maximum value in the array
norm_flat1B_bias = flat1B_bias / np.max(flat1B_bias)
norm_flat1R_bias = flat1R_bias / np.max(flat1R_bias)
norm_flat2B_bias = flat2B_bias / np.max(flat2B_bias)
norm_flat2R_bias = flat2R_bias / np.max(flat2R_bias)

# Open the first light image to use its header for WCS (World Coordinate System) information
light1B_ref_img = fits.open(light1B_files[0])
light1B_ref_header = light1B_ref_img[0].header
ref_RA = light1B_ref_header["RA"]
ref_DEC = light1B_ref_header["DEC"]
light1B_ref_data = light1B_ref_img[0].data

# Correct light images using dark images and normalized flat images
HDU1B_list = helper.make_header_data_lists(light1B_files, darks_img, norm_flat1B_bias)
HDU1R_list = helper.make_header_data_lists(light1R_files, darks_img, norm_flat1R_bias)
HDU2B_list = helper.make_header_data_lists(light2B_files, darks_img, norm_flat2B_bias)
HDU2R_list = helper.make_header_data_lists(light2R_files, darks_img, norm_flat2R_bias)

# Align the corrected light images to the reference image
lightB_master = helper.align_image(HDU1B_list + HDU2B_list, light1B_ref_data)
lightR_master = helper.align_image(HDU1R_list + HDU2R_list, light1B_ref_data)

# Create FITS files for the aligned master images using the reference header for WCS
hdul1B = fits.PrimaryHDU(lightB_master, light1B_ref_header)
hdul1R = fits.PrimaryHDU(lightR_master, light1B_ref_header)

# Define output filenames for the aligned master images
outfile1 = DIRECTORY + "Light_Nov2_B.fits"
outfile2 = DIRECTORY + "Light_Nov2_R.fits"

# Write the aligned master images to the output files, overwriting existing files if necessary
hdul1B.writeto(outfile1, overwrite=True)
hdul1R.writeto(outfile2, overwrite=True)

# Open the output FITS files for plotting and further analysis
b_filter = fits.open(outfile1)
r_filter = fits.open(outfile2)

# Plot the blue filter image and save it
plt.imshow(b_filter[0].data, cmap='gray', norm=LogNorm(vmin=1, vmax=100000))
plt.savefig(DIRECTORY + "b_filter_image.png")
plt.clf()

# Plot the red filter image and save it
plt.imshow(r_filter[0].data, cmap='gray', norm=LogNorm(vmin=1, vmax=100000))
plt.savefig(DIRECTORY + "r_filter_image.png")
plt.clf()

# Extract a subset of the data for detailed analysis
b_data = b_filter[0].data[1000:1400, 2300:2700]
r_data = r_filter[0].data[1000:1400, 2300:2700]

# Plot and save the subset of blue filter data
plt.imshow(b_data, cmap=cm.gray, norm=LogNorm(vmin=1, vmax=100000))
plt.savefig(DIRECTORY + "subset_b.png")
plt.clf()

# Plot and save the subset of red filter data
plt.imshow(r_data, cmap=cm.gray, norm=LogNorm(vmin=1, vmax=100000))
plt.savefig(DIRECTORY + "subset_r.png")
plt.clf()

# Calculate statistics (mean, median, and standard deviation) for the subsets of blue and red filter data
b_mean, b_median, b_std = sigma_clipped_stats(b_data, sigma=3.0)
r_mean, r_median, r_std = sigma_clipped_stats(r_data, sigma=3.0)

# Print the calculated statistics
print(np.array((b_mean, b_median, b_std)))
print(np.array((r_mean, r_median, r_std)))

# Detect stars in the blue filter subset using DAOStarFinder
daofind = DAOStarFinder(fwhm=5.0, threshold=3. * b_std)
b_sources = daofind(b_data - b_median)

# Detect stars in the red filter subset using DAOStarFinder
r_sources = daofind(r_data - r_median)

# Format the output for better readability
for col in b_sources.colnames:
    if col not in ('id', 'npix'):
        b_sources[col].info.format = '%.2f'  # for consistent table output

for col in r_sources.colnames:
    if col not in ('id', 'npix'):
        r_sources[col].info.format = '%.2f'

# Plot detected star positions on the blue filter subset image
positions = np.transpose((b_sources['xcentroid'], b_sources['ycentroid']))
apertures = SkyCircularAperture(positions, r=6.0)
plt.imshow(b_data, cmap=cm.gray, origin='lower', norm=LogNorm(vmin=10, vmax=1000), interpolation='nearest')
apertures.plot(color='red', lw=1.5, alpha=0.5)
plt.savefig(DIRECTORY + "detected_b_stars.png")
plt.clf()

# Plot detected star positions on the red filter subset image
positions = np.transpose((r_sources['xcentroid'], r_sources['ycentroid']))
apertures = SkyCircularAperture(positions, r=6.0)
plt.imshow(r_data, cmap=cm.gray, origin='lower', norm=LogNorm(vmin=10, vmax=1000), interpolation='nearest')
apertures.plot(color='red', lw=1.5, alpha=0.5)
plt.savefig(DIRECTORY + "detected_r_stars.png")
plt.clf()

# Find peaks in the blue and red filter images
b_xy = find_peaks(b_filter[0].data)[0:20]
r_xy = find_peaks(r_filter[0].data)[0:20]

# Plot peaks detected in the blue filter image and save the plot
plt.imshow(b_filter[0].data, vmin=0.1*np.median(b_filter[0].data), vmax=5 * np.median(b_filter[0].data), cmap="Greys_r")
_ = CircularAperture(b_xy, r=10.0).plot(color="m")
plt.savefig(DIRECTORY + "blue_filter_peaks.png")
plt.clf()

# Plot peaks detected in the red filter image and save the plot
plt.imshow(r_filter[0].data, vmin=0.1*np.median(r_filter[0].data), vmax=5 * np.median(r_filter[0].data), cmap="Greys_r")
_ = CircularAperture(r_xy, r=10.0).plot(color="m")
plt.savefig(DIRECTORY + "red_filter_peaks.png")
plt.clf()

# Compute World Coordinate System (WCS) for the blue filter image
true_wcs = WCS(b_filter[0].header)
fov = (b_filter[0].data.shape * proj_plane_pixel_scales(true_wcs))[0]

# Compute World Coordinate System (WCS) for the red filter image
true_wcs = WCS(r_filter[0].header)
fov = (r_filter[0].data.shape * proj_plane_pixel_scales(true_wcs))[0]

# Calculate statistics (mean, median, and standard deviation) for the blue and red filter images
b_mean, b_median, b_std = sigma_clipped_stats(b_filter[0].data, sigma=3.0)
r_mean, r_median, r_std = sigma_clipped_stats(r_filter[0].data, sigma=3.0)

# # Commented out code for obtaining RA and DEC of stars from Gaia catalog and sparsifying them
# all_radecs = gaia_radecs((b_filter[0].header['RA'], b_filter[0].header['DEC']), 1.2 * (4784/2*0.98/60/60))
# all_radecs = sparsify(all_radecs, 0.01)

# Obtain RA and DEC of stars from Gaia catalog with a fixed reference position and sparsify them
all_radecs = gaia_radecs((323.37641682792, -0.779124132264951), 1.2 * 0.65)
all_radecs = sparsify(all_radecs, 0.01)

# Compute WCS for the blue and red filter images using detected peaks and Gaia catalog positions
b_wcs = compute_wcs(b_xy, all_radecs[0:30], tolerance=10)
r_wcs = compute_wcs(r_xy, all_radecs[0:30], tolerance=10)

# Plot and check the WCS alignment for the blue filter image
radecs_xy = np.array(b_wcs.world_to_pixel_values(all_radecs))
plt.imshow(b_filter[0].data, vmin=np.median(b_filter[0].data), vmax=3 * np.median(b_data), cmap="Greys_r")
_ = CircularAperture(radecs_xy[0:30], 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "check_wcs_b.png")
plt.clf()

# Plot and check the WCS alignment for the red filter image
radecs_xy = np.array(r_wcs.world_to_pixel_values(all_radecs))
plt.imshow(r_filter[0].data, vmin=np.median(r_filter[0].data), vmax=3 * np.median(b_data), cmap="Greys_r")
_ = CircularAperture(radecs_xy[0:30], 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "check_wcs_r.png")
plt.clf()

# Update the headers of the blue and red filter images with the computed WCS
b_filter[0].header.update(b_wcs.to_header())
r_filter[0].header.update(r_wcs.to_header())

# Normalize the blue and red filter images and save them with the updated WCS
b_filter[0].data = b_filter[0].data / (16 * 120.0)
b_filter[0].writeto(DIRECTORY + "b_w_WCS.fits", overwrite=True)

r_filter[0].data = r_filter[0].data / (16 * 120.0)
r_filter[0].writeto(DIRECTORY + "r_w_WCS.fits", overwrite=True)

# Read positional and photometric data from files
coords = pd.read_csv("NGC7089.pos", sep='[ ]{2,}', engine='python')
mags = pd.read_csv("NGC7089.pho", sep='[ ]{1,}', engine='python')

# Rename columns for clarity and convert DEC to float
coords.rename(columns={'323.36209232807': 'RA', '-00.84555554920': 'DEC'}, inplace=True)
coords.DEC = coords.DEC.str.replace(" ", "")
coords.DEC = coords.DEC.astype(float)

# Merge positional and photometric data, filtering stars with specific magnitude criteria
stetson = pd.merge(coords[["Reference", 'RA', 'DEC']], mags, how='inner', on=['Reference'])
stetson = stetson[(stetson.B < 19) & (stetson.R < 19)]

# Plot Stetson Photometric Standard Star positions on the blue filter image and save the plot
b_stetson_pos = np.array(b_wcs.world_to_pixel_values(stetson[["RA", "DEC"]]))
ax = plt.subplot(projection=b_wcs)
ax.imshow(b_filter[0].data, vmin=np.median(b_filter[0].data), vmax=3 * np.median(b_filter[0].data), cmap="Greys_r")
_ = CircularAperture(b_stetson_pos, 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "stetson_stars_b.png")
plt.clf()

# Plot Stetson Photometric Standard Star positions on the red filter image and save the plot
r_stetson_pos = np.array(r_wcs.world_to_pixel_values(stetson[["RA", "DEC"]]))
ax = plt.subplot(projection=r_wcs)
ax.imshow(r_filter[0].data, vmin=np.median(r_filter[0].data), vmax=3 * np.median(r_filter[0].data), cmap="Greys_r")
_ = CircularAperture(r_stetson_pos, 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "stetson_stars_r.png")
plt.clf()

# Create cutouts from the blue and red filter images for focused analysis
position = (2200, 1000)
size = (1000, 1000)
b_cutout = Cutout2D(b_filter[0].data, position, size, wcs=b_wcs)
r_cutout = Cutout2D(r_filter[0].data, position, size, wcs=r_wcs)
b_analysis = fits.PrimaryHDU()
r_analysis = fits.PrimaryHDU()

# Update the cutouts with their respective WCS headers
b_analysis.data = b_cutout.data
b_analysis.header.update(b_cutout.wcs.to_header())

r_analysis.data = r_cutout.data
r_analysis.header.update(r_cutout.wcs.to_header())

# Find peaks in the blue cutout image and compute WCS
b_xy_cutout = find_peaks(b_analysis.data)[0:20]
b_wcs_cutout = compute_wcs(b_xy_cutout, all_radecs[0:30], tolerance=10)

# Find peaks in the red cutout image and compute WCS
r_xy_cutout = find_peaks(r_analysis.data)[0:20]
r_wcs_cutout = compute_wcs(r_xy_cutout, all_radecs[0:30], tolerance=10)

# Convert Stetson photometric standard star positions to pixel coordinates for the blue filter cutout
b_stetson_pos = np.array(b_wcs_cutout.world_to_pixel_values(stetson[["RA", "DEC"]]))

# Create a subplot with WCS projection for the blue filter analysis
ax = plt.subplot(projection = b_wcs)
plt.imshow(b_analysis.data, vmin=np.median(b_filter[0].data), vmax=3 * np.median(b_filter[0].data), cmap="Greys_r")

# Plot the Stetson star positions on the blue filter cutout image
_ = CircularAperture(b_stetson_pos, 10).plot(color="m", alpha=0.5)

# Convert Stetson photometric standard star positions to pixel coordinates for the red filter cutout
r_stetson_pos = np.array(r_wcs_cutout.world_to_pixel_values(stetson[["RA", "DEC"]]))

# Create a subplot with WCS projection for the red filter analysis
ax = plt.subplot(projection= r_wcs)
plt.imshow(r_analysis.data, vmin=np.median(r_filter[0].data), vmax=3 * np.median(r_filter[0].data), cmap="Greys_r")

# Plot the Stetson star positions on the red filter cutout image
_ = CircularAperture(r_stetson_pos, 10).plot(color="m", alpha=0.5)

# Calculate statistics (mean, median, and standard deviation) for the blue and red analysis cutout data
b_mean_new, b_median_new, b_std_new = sigma_clipped_stats(b_analysis.data, sigma=3.0)
r_mean_new, r_median_new, r_std_new = sigma_clipped_stats(r_analysis.data, sigma=3.0)

# Initialize DAOStarFinder for the blue and red filter cutout data with specified FWHM and threshold
b_daofind = DAOStarFinder(fwhm=7.0, threshold=5.*b_std)
r_daofind = DAOStarFinder(fwhm=7.0, threshold=5.*r_std)

# Detect stars in the blue filter analysis cutout data
b_sources = b_daofind(b_analysis.data - b_median_new)
for col in b_sources.colnames:  
    if col not in ('id', 'npix'):
        b_sources[col].info.format = '%.2f'  # Format output for consistency

# Detect stars in the red filter analysis cutout data
r_sources = r_daofind(r_analysis.data - r_median_new)
for col in r_sources.colnames:  
    if col not in ('id', 'npix'):
        r_sources[col].info.format = '%.2f'

# Convert source detection results to pandas DataFrames
b_detected_table = b_sources.to_pandas()
r_detected_table = r_sources.to_pandas()

# Filter detected sources by flux threshold
b_detected_table = b_detected_table[(b_detected_table.flux > 0.79)]
r_detected_table = r_detected_table[(r_detected_table.flux > 0.79)]

# Extract positions of detected sources for the blue and red filter analysis
b_det_pos = np.transpose((b_detected_table['xcentroid'], b_detected_table['ycentroid']))
r_det_pos = np.transpose((r_detected_table['xcentroid'], r_detected_table['ycentroid']))

# Convert Stetson star positions to pixel coordinates for plotting on the analysis cutouts
b_stetson_pos = np.array(b_wcs_cutout.world_to_pixel_values(stetson[["RA", "DEC"]]))
r_stetson_pos = np.array(r_wcs_cutout.world_to_pixel_values(stetson[["RA", "DEC"]]))

# Plot the blue filter analysis cutout with Stetson stars and detected source positions
plt.imshow(b_analysis.data, vmin=np.median(b_filter[0].data), vmax=3 * np.median(b_filter[0].data), cmap="Greys_r")
_ = CircularAperture(b_stetson_pos, 10).plot(color="g", alpha=0.5)
_ = CircularAperture(b_det_pos, 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "blue_analysis_stars.png")
plt.clf()

# Plot the red filter analysis cutout with Stetson stars and detected source positions
plt.imshow(r_analysis.data, vmin=np.median(r_filter[0].data), vmax=3 * np.median(r_filter[0].data), cmap="Greys_r")
_ = CircularAperture(r_stetson_pos, 10).plot(color="g", alpha=0.5)
_ = CircularAperture(r_det_pos, 10).plot(color="m", alpha=0.5)
plt.savefig(DIRECTORY + "red_analysis_stars.png")
plt.clf()

# Convert detected source positions to world coordinates (RA/DEC) and add to DataFrames
b_detected_table['coords'] = b_wcs_cutout.pixel_to_world(b_detected_table['xcentroid'], b_detected_table['ycentroid'])
r_detected_table['coords'] = r_wcs_cutout.pixel_to_world(r_detected_table['xcentroid'], r_detected_table['ycentroid'])

# Create a subplot with WCS projection for the blue filter analysis
b_ax = plt.subplot(projection=b_wcs)
# Display the blue filter analysis cutout image
plt.imshow(b_analysis.data, vmin=np.median(b_filter[0].data), vmax=3 * np.median(b_filter[0].data), cmap="Greys_r")
# Plot Stetson star positions and detected star positions on the blue filter image
_ = CircularAperture(b_stetson_pos, 5).plot(color="g", alpha=0.5)
_ = CircularAperture(b_det_pos, 5).plot(color="m", alpha=0.5)
# Save the plot
plt.savefig(DIRECTORY + "blue_analysis_with_stars.png")
plt.clf()

# Create a subplot with WCS projection for the red filter analysis
r_ax = plt.subplot(projection=r_wcs)
# Display the red filter analysis cutout image
plt.imshow(r_analysis.data, vmin=np.median(r_filter[0].data), vmax=3 * np.median(r_filter[0].data), cmap="Greys_r")
# Plot Stetson star positions and detected star positions on the red filter image
_ = CircularAperture(r_stetson_pos, 5).plot(color="g", alpha=0.5)
_ = CircularAperture(r_det_pos, 5).plot(color="m", alpha=0.5)
# Save the plot
plt.savefig(DIRECTORY + "red_analysis_with_stars.png")
plt.clf()

# Convert Stetson positions to SkyCoord objects
c_stetson = SkyCoord(stetson.RA, stetson.DEC, unit=u.degree)

# Convert detected star positions to world coordinates for blue and red filters
b_det_radec = b_wcs_cutout.pixel_to_world(b_detected_table['xcentroid'], b_detected_table['ycentroid'])
c_det_b_filter = SkyCoord(b_det_radec.ra, b_det_radec.dec)

r_det_radec = r_wcs_cutout.pixel_to_world(r_detected_table['xcentroid'], r_detected_table['ycentroid'])
c_det_r_filter = SkyCoord(r_det_radec.ra, r_det_radec.dec)

# Match detected stars to Stetson catalog for blue and red filters
idx_b_filter, d2d_b_filter, d3d_b_filter = c_det_b_filter.match_to_catalog_sky(c_stetson)
idx_r_filter, d2d_r_filter, d3d_r_filter = c_det_r_filter.match_to_catalog_sky(c_stetson)

# Add a SkyCoord column to the Stetson DataFrame
stetson['sky'] = SkyCoord(stetson.RA, stetson.DEC, unit=u.degree)

# Define the maximum separation for matches
max_sep = 1.0 * u.arcsec
# Apply separation constraint for matches
b_sep_constraint = d2d_b_filter < max_sep
r_sep_constraint = d2d_r_filter < max_sep

# Get matching coordinates for blue and red filters
c_matches_b_filter = c_det_b_filter[b_sep_constraint]
c_matches_r_filter = c_det_r_filter[r_sep_constraint]

# Get matching catalog entries for blue and red filters
b_catalog_matches = c_stetson[idx_b_filter[b_sep_constraint]]
r_catalog_matches = c_stetson[idx_r_filter[r_sep_constraint]]

# Add SkyCoord objects to detected star DataFrames
b_detected_table['coords'] = c_det_b_filter
r_detected_table['coords'] = c_det_r_filter

# Extract magnitudes of detected stars that match catalog entries for blue filter
b_det_array = []
for i in b_detected_table.index:
    for j in range(0, len(c_matches_b_filter)):
        if b_detected_table.coords[i] == c_matches_b_filter[j]:
            b_det_array.append(b_detected_table.mag[i])
# Magnitudes are calculated as -2.5 * log10(flux)

# Extract magnitudes of detected stars that match catalog entries for red filter
r_det_array = []
for i in r_detected_table.index:
    for j in range(0, len(c_matches_r_filter)):
        if r_detected_table.coords[i] == c_matches_r_filter[j]:
            r_det_array.append(b_detected_table.mag[i])
# Magnitudes are calculated as -2.5 * log10(flux)

# Extract calibrated magnitudes from the Stetson catalog for blue filter
b_calib_array = []
for i in stetson.index:
    for j in range(0, len(b_catalog_matches)):
        if(stetson.sky[i] == b_catalog_matches[j]):
            b_calib_array.append(stetson.B[i])

# Extract calibrated magnitudes from the Stetson catalog for red filter
r_calib_array = []
for i in stetson.index:
    for j in range(0, len(r_catalog_matches)):
        if(stetson.sky[i] == r_catalog_matches[j]):
            r_calib_array.append(stetson.B[i])

# Create a scatter plot of detected vs calibrated magnitudes and save the plot
plt.scatter(b_det_array, b_calib_array, label='Blue Filter')
plt.scatter(r_det_array, r_calib_array, label='Red Filter', color='r')
plt.xlabel('Detected Magnitude')
plt.ylabel('Calibrated Magnitude')
plt.legend()
plt.savefig(DIRECTORY + "magnitude_comparison.png")
plt.clf()
