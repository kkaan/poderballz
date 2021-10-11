# -*- coding: utf-8 -*-
"""
Calculate Winston Luts results for multiple targets in a Dynamic HTT.

Used alongside the a phantom with multiple ball bearing targets and an MLC 
based plan with DCAT like apertures centred on the targets. The analysis will 
provide details on the accuracy with which the planned apetures are delivered
on treatment for a VMAT field with dynamic MLC and gantry rotation.

.. module:: poderballz
   :platform: Windows
   :synopsis: grab some balls.

.. moduleauthor:: Kaan Kankean <kankean.kandasamy@health.nsw.gov.au>

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas
import glob
import os
import cv2
import imageio

from pathlib import Path
from operator import itemgetter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from poderprogressbar import update_progress
from poderPlot import plot_against_gantry, boxplot, plot_coords_on_images
from poderInterpolate import poly_interpolate


def sparse_image(img,cropx,cropy):
    """
    Crop the imge and make it sparse.
    
    Parameters
    ----------
    img : numpy array
        image to make sparse.
    cropx : int
        number pixels in the y direction.
    cropy : int
        numnber of pixels in the x direction.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_apeture_centroids(image, binary, num_of_balls):
    """
    Calculate the locations of all the mlc defined aperture positions.
    
    The location will be returned as the coordinates of the centre 
    of mass of the aperture.
    
    Parameters
    ----------
    image : numpy array
        image within which to find the mlc apertures.
    binary : boolean array
        boolean mask for epid image to only include apetures

    Returns
    -------
    centroids : list of tuples
        list of apertures found in increasing y coordingates.
        
    """
    label_image = label(binary)
    apertures = regionprops(label_image)
    apertures.sort(key=lambda a: a.convex_area, reverse=True)
    apertures = apertures[: num_of_balls] #just grab the largest apertures
    centroids = [a.centroid for a in apertures] 
    centroids = sorted(centroids, key=lambda a: a[0]) #sort by y value
        
    centroids = [(sub[1], sub[0]) for sub in centroids]
    
    return centroids

def get_ball_positions(image, binary, num_of_balls):
    """
    Return the positions of all the balls found in the image.
    
    This function is used by the funtions get_drr_balls and get_epid_balls.

    Parameters
    ----------
    sel : numy_array of binarised imaged
        DESCRIPTION.

    Returns
    -------
    ball_positions : TYPE
        DESCRIPTION.

    """
    edges = canny(image, sigma=3, low_threshold=5, high_threshold=10, mask=binary)
    hough_radii = np.arange(4, 15)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=num_of_balls)
    ball_positions = list(zip(cx,cy))
    
    # order balls by y position.
  
    ball_positions = sorted(ball_positions, key=itemgetter(1)) 
    
    #TODO - we will have to generalise this for geometry agnostic behaviour
    return ball_positions

def get_epid_balls_and_apertures(names, num_of_balls):
    """
    Get the ball and apeture positions from the EPID image.

    Parameters
    ----------
    files : list of strings
        contains the names of tif image files.

    Returns
    -------
    None.

    """
    for i, n in enumerate(names):
        filename = epidfolder / n
        im = imageio.imread(filename)
        im = np.array(im)
        
        im = sparse_image(im, cropx, cropy)
        thresh = threshold_otsu(im)
        binary = im > thresh
        sel = np.zeros_like(im)
        sel[binary] = im[binary]
    
        apeture_centroids = get_apeture_centroids(sel, binary, num_of_balls)
        apeture_centroids = [item for t in apeture_centroids for item in t]
        apeture_centroids = [int(item) for item in apeture_centroids]
   
        ball_positions = get_ball_positions(sel, binary, num_of_balls)
        ball_positions = [item for t in ball_positions for item in t]
    
    
        try:
            df.at[i, 'EPIDApertures'] = apeture_centroids
            df.at[i, 'EPIDBalls'] = ball_positions
                # df.at[] is faster than df.loc[] but will preserve data 
                # type of df series. And it will do it silently. Saving 
                # floats in int columns will be lossful.
        except AssertionError as error:
            print(error)
            print("Probably found too many balls or apertures." +
                  "Change detection settings")
        
        #current status
        text = "Finding balls and apertures in EPID {0}".format(n)
        
        # Progress bar
        update_progress(i/progmax, text)
      
def get_drr_balls(names, num_of_balls):
    """
    Get the ball positions from the drr image.

    Parameters
    ----------
    files : list of strings
        contains the names of tif image files.

    Returns
    -------
    None.

    """
    for i, n in enumerate(names):
        filename = drrfolder / n
        im = imageio.imread(filename)
        im = np.array(im)
        
        im = sparse_image(im, cropx, cropy)
        thresh = threshold_otsu(im)
        binary = im > thresh
        sel = np.zeros_like(im)
        sel[binary] = im[binary]
    
        ball_positions = get_ball_positions(sel, binary, num_of_balls)
        ball_positions = [item for t in ball_positions for item in t]
    
    
        try:
            df.at[i, 'DRRBalls'] = ball_positions
                # df.at[] is faster than df.loc[] but will preserve data 
                # type of df series. And it will do it silently. Saving 
                # floats in int columns will be lossful.
        except AssertionError as error:
            print(error)
            print("Probably found too many balls or apertures." + 
                  "Change detection settings")
        
        # Progress bar
        
        text = "Finding balls in DRR {0}".format(n)
        update_progress(i/progmax, text)

def get_drr_apertures(names, num_of_balls):
    """
    Get apeture positions from the mlc image and saves to dataframe.

    Parameters
    ----------
    names : list of strings
        contains the names of tif image files.
    num_of_balls: integer
        how many balls do you have?

    Returns
    -------
    None.

    """
    for i, n in enumerate(names):
        filename = mlcfolder / n
        
        im = imageio.imread(filename)
        
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask = cv2.morphologyEx(im, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        
        #mask = np.dstack([mask, mask, mask]) / 255
        im = im * mask
        
        #TODO: Check if the above removal of leaf gap also narrows the objects.

        # cv2.imshow('Output', out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        im = np.array(im)
        
        im = sparse_image(im, cropx, cropy)
        thresh = threshold_otsu(im)
        binary = im > thresh
        sel = np.zeros_like(im)
        sel[binary] = im[binary]
    
        aperture_centroids = get_apeture_centroids(sel, binary, num_of_balls)
        aperture_centroids = [item for t in aperture_centroids for item in t]
        aperture_centroids = [int(item) for item in aperture_centroids]
   
        try:
            df.at[i, 'DRRApertures'] = aperture_centroids
                # df.at[] is faster than df.loc[] but will preserve data 
                # type of df series. And it will do it silently. Saving 
                # floats in int columns will be lossful.
        except AssertionError as error:
            print(error)
            print("Probably found too many balls or apertures." +
                  "Change detection settings")
        
        text = "\rProcessing DRRs, find apertures in {0}".format(n)
        sys.stdout.write(text)
        sys.stdout.flush()
        
        text = "Finding apertures in MLC {0}".format(n)
        update_progress(i/progmax, text)

def calculateWL(df):
    """
    Calculate Winston Lutz deviations and saves to dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe of ball and aperture details.

    Returns
    -------
    None.

    """
    # calculate deviation in mm
    # Resolution details in DICOM header Image plane pixel spacing (3002,0011)
    # 0.336 mm/px at SID 100cm
    
    
    # SDD = 500 #mm from iso
    # pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    # pixel_to_mm = pixel_res*(1000+SDD)/1000
    
    pixel_to_mm = 1 # conversion is now done prior to interpolation.
       
    df['WL'] = pixel_to_mm*(df.loc[:, 'DRRBalls'] -df.loc[:, 'EPIDBalls'])-(
        pixel_to_mm*(df.loc[:, 'DRRApertures'] - df.loc[:, 'EPIDApertures']))
    
    
    
    # remove values higher than 5 mm 
    df['WL'] = df['WL'][np.abs(df.WL) < 5]

if __name__ == "__main__": 
    
    fstring = 'P:/14 Projects/49_SRS Phantom/Output Images'
    data_folder = ('P:/14 Projects/49_SRS Phantom/Output Images')
    data_folder = Path(data_folder)
    frameinfo = data_folder / 'Gantry_Angles.csv'
    epidfolder = data_folder / 'EPID'
    drrfolder = data_folder / 'DRR'
    mlcfolder = data_folder / 'MLC'
    
    #create dataframe with gantry angles and filenames
    
    item_type = ["EPIDBalls","EPIDApertures","DRRBalls","DRRApertures", "WL"]
    num_of_balls = 4
    item_number = list(range(1,num_of_balls+1))
    axes = ['x', 'y']
    mi = pandas.MultiIndex.from_product([item_type, item_number, axes])
    df = pandas.DataFrame(columns=mi)
    gdf=pandas.read_csv(frameinfo, header=None, names=["Gantry"])
    df['Gantry']= gdf['Gantry']
    
    
    # #some extra formatting of the dataframe
    # image_df['EPIDBalls'] = image_df['EPIDBalls'].astype(object)
    # image_df['EPIDApertures'] = image_df['EPIDApertures'].astype(object)
    
            
    #load epid image names
    
    names = [os.path.basename(x) for x in glob.glob(fstring+'/EPID/*.tif')]
    df['filename'] = names
    
    progmax = len(df)-1
    
    cropx = 900
    cropy = 900
    
    #Process all images and save ball and aperture positions.
    get_epid_balls_and_apertures(names, num_of_balls)
    get_drr_balls(names, num_of_balls)
    get_drr_apertures(names, num_of_balls)
    
    plot_against_gantry('DRRApertures', num_of_balls, df) # EPIDBalls, EPIDApertures, DRRBalls, DRRAperturs
    plt.show()
    plot_against_gantry('DRRBalls', num_of_balls, df)
    plt.show()
    plot_against_gantry('EPIDApertures', num_of_balls, df)
    plt.show()
    plot_against_gantry('EPIDBalls', num_of_balls, df)
    plt.show()
    
    # Allowed arguments:'drrballs', 'drrape' 'epid' 
    plot_coords_on_images('epid', range(28,53), data_folder, df)
    plot_coords_on_images('drrape', range(28,53), data_folder, df)
    plot_coords_on_images('drrballs', range(28,53), data_folder, df)

    poly_interpolate(df)
    calculateWL(df)
    
    plot_coords_on_images('epid', range(28,53), data_folder, df) 
    plt.show()
        
   
       
    

        
    
    plot_against_gantry('WL', num_of_balls, df)
    plt.show()
    boxplot('WL', num_of_balls, df)
    plt.show()

    plt.close('all') # memory save


    


















