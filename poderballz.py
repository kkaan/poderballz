# -*- coding: utf-8 -*-
"""
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
import imageio

from pathlib import Path
from skimage import color
from operator import itemgetter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu

import cv2

import scipy.stats as stats


def get_apeture_centroids(image, binary, num_of_balls):
    """
    Calculate the locations of all the mlc defined aperture positions in
    the image. The location will be returned as the coordinates of the centre 
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
    '''
    Return the positions of all the balls found in the image.

    Parameters
    ----------
    sel : numy_array of binarised imaged
        DESCRIPTION.

    Returns
    -------
    ball_positions : TYPE
        DESCRIPTION.

    '''
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



def sparse_image(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


#progress bar
def update_progress(progress, subtext):
    '''
    Creates a progress bar in standard output.

    Parameters
    ----------
    progress : int, float
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2} {3}".format( "#"*block + 
                                                  "-"*(barLength-block),
                                                  int(progress*100),
                                                  status,
                                                  subtext)
    sys.stdout.write(text)
    sys.stdout.flush()
    

# Plotting the balls
def plot_against_gantry(what):
    """
    
    Plots ball positions over gantry agnles
    

    Returns
    -------
    None.

    """

    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('PoderBallz A tale of four balls {}'.format(what))
    
    colours = ['darkred', 'darkblue', 'darkgreen', 'darkslategrey']
    # need more colours for more balls.
    
    # num_of_balls = 1 # comment out this line to print all balls
    
    for i in range(1,num_of_balls+1):
        df.plot(kind="scatter", x='Gantry', y=(what, i, 'x'), s=1,
                color=colours[i-1], label='ball {}'.format(i), ax=ax1)
        
        df.plot(kind="scatter", x='Gantry', y=(what, i, 'y'), s=1,
                color=colours[i-1], label='ball {}'.format(i), ax=ax2)
    
    ax1.set_ylabel('X position (mm)')  
    ax2.set_ylabel('Y position (mm)')
    
    ax1.legend(title="None", fontsize="xx-small", loc="upper right")
    ax2.get_legend().remove()
    
    
    plt.show()
    
    

def plot_coords_on_images(what, range_of_images):
    """
    
    Plots the coordinates of balls and apertures on the images.
    
    Parameters
    ----------
    image : numpy array
        EPID image on which to show ball positions.
    apertures : list of tuples
        apeture positions.
    balls : list of tuples 
        ball positions.

    Returns
    -------
    None.

    """
    if what == 'drrballs':
        folder = drrfolder
        aperture_dflabel = 'DRRApertures'
        balls_dflabel = 'DRRBalls'
    elif what == 'drrape':
        folder = mlcfolder
        aperture_dflabel = 'DRRApertures'
        balls_dflabel = 'DRRBalls'
    elif what == 'epid':
        folder = epidfolder
        aperture_dflabel = 'EPIDApertures'
        balls_dflabel = 'EPIDBalls'
        
    
    
    for i in range_of_images:
        # test plotting to see if the coordinates makes sense
        filename = df.loc[i, 'filename'].values[0]
        filename = folder / filename
        im = imageio.imread(filename)
        im = np.array(im)
        
        if what == 'epid':
            balls = df.loc[i, balls_dflabel].values
            apertures = df.loc[i, aperture_dflabel].values
        elif what == 'drr' or 'drrape':
            balls = df.loc[i, balls_dflabel].values
            apertures = df.loc[i, aperture_dflabel].values

        
        #convert to format required by plotting function
        it = iter(balls)
        balls = [*zip(it,it)]
        
        it = iter(apertures)
        apertures = [*zip(it,it)]
        image = sparse_image(im, 900, 900)
        
        
        fig, ax = plt.subplots()
        image = color.gray2rgb(image)
    
        
        for a in apertures:
                ax.plot(a[0], a[1], color="darkred", marker='x', 
                        linewidth=3, markersize=5)
        
        for b in balls:
                ax.plot(b[0], b[1], color="darkblue",marker='o', 
                        linewidth=3, markersize=2)
        
        ax.imshow(image)
        ax.title.set_text(filename)
        plt.show()
        
def plot_box():    
    data_1 = (df['WL', 1,'x']**2*df['WL', 1,'y']**2)**(1/2)
    data_2 = (df['WL', 2,'x']**2*df['WL', 2,'y']**2)**(1/2)
    data_3 = (df['WL', 3,'x']**2*df['WL', 3,'y']**2)**(1/2)
    data_4 = (df['WL', 4,'x']**2*df['WL', 4,'y']**2)**(1/2) 
    
    data = [data_1, data_2, data_3, data_4]
    fig = plt.figure(figsize =(7, 5))
     
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
     
    # Creating plot
    bp = ax.boxplot(data)
    ticks = plt.xticks([1,2,3,4],[30,50,60,100])
    plt.title(label="MTSI Positional Verification", fontsize=25)
    plt.xlabel("Distance from Isocentre (mm)", fontsize=10)
    plt.ylabel("Deviation from DRR (mm)", fontsize=10)
    
    # show plot
    plt.show()

    
def get_epid_balls_and_apertures(names, num_of_balls):
    """
    Gets the ball and apeture positions from the EPID image.

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
    Gets the ball positions from the drr image.

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
    Gets apeture positions from the mlc image.

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


def calculateWL():
    # calculate deviation in mm
    # Resolution details in DICOM header Image plane pixel spacing (3002,0011)
    # 0.336 mm/px at SID 100cm
    
    
    SDD = 500 #mm from iso
    pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    pixel_to_mm = pixel_res*(1000+SDD)/1000
    
       
    df['WL'] = pixel_to_mm*(df.loc[:, 'DRRBalls'] -df.loc[:, 'EPIDBalls'])-(
        pixel_to_mm*(df.loc[:, 'DRRApertures'] - df.loc[:, 'EPIDApertures']))
    
    
    # remove the extremes
    # df['WL'] = df['WL'][np.abs(df.WL-df.WL.mean()) <= (1*df.WL.std())]
    
    # remove values higher than 5 mm 
    df['WL'] = df['WL'][np.abs(df.WL) < 5]
    
    plot_against_gantry('WL')


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
    
    plot_coords_on_images('epid', range(28,53)) 
    # Allowed arguments:'drrballs', 'drrape' 'epid' 
        
    
    plot_against_gantry('DRRApertures') # EPIDBalls, EPIDApertures, DRRBalls, DRRAperturs
    plot_against_gantry('DRRBalls')
    
    plot_against_gantry('EPIDApertures')
    plot_against_gantry('EPIDBalls')
    
    calculateWL
    # Scratch

    plot_box()

    #branch testing

























