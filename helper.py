#for math
import numpy as np

from astropy.io import fits

# for directory
import os

# fo rallighining
import astroalign as align


#ASTR 3070 Data Reduction
##########################################################################################################################

#Make a function to get the filenames so then I can open the fits files.

def get_filename(directory):
    '''
    This gives all the filenames that are in a folder and returns them in list format
    '''
    filename_list=[]
    for filename in os.scandir(directory):
        if os.path.isfile(filename):
            filename_list.append(filename)
    return(filename_list)



def make_master(file_list):
    '''
    This takes a list of filenames
    Loads the data, averages all the images together
    Returns a master image
    '''
    imgs1=fits.open(file_list[0])
    data=imgs1[0].data

    height=len(data[:,0])
    width=len(data[0,:])
    
    master_img=np.zeros((height, width))
    count=0
    for file in file_list:
        count+=1
        imgs=fits.open(file)
        img_data=imgs[0].data
        master_img=master_img+img_data

    master_img=master_img/count

    return master_img


def make_header_data_lists(light, dark, norm_flats_bias):
    '''
    light = list of light file names
    dark = master dark image
    norm_flats_bias = normalized bias subtracted flat images

    returns - list of image HDUs
    '''
    corr_HDU_list=[]
    for file in light:
        light_imgs=fits.open(file)
        light_data=light_imgs[0].data
        #light_header=light_imgs[0].header
        light_corr=(light_data-dark)/norm_flats_bias
        corr_HDU_list.append(light_corr)
    return corr_HDU_list



#Do astroalign and then use astroquery to get RA and DEC


def align_image(list_of_images, reference_image):

 
    height=len(reference_image[:,0])
    width=len(reference_image[0,:])
    Master_image=np.zeros((height, width))
    count=0

    for image in list_of_images:
        count+=1
        aligned_image, footprint=align.register(image, reference_image)
        Master_image=Master_image+aligned_image

    Master_image=Master_image/count

    return Master_image



