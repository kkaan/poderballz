# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:19:56 2021

Process images and DRRs from the PoderPhan VMAT Hidden Target Tests
Output the deviations in measured positions from the DRR positions.



@author: kaan
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


def get_apeture_centroids(image, binary, number_of_balls
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
    apertures = apertures[: number_of_balls]
    centroids = [a.centroid for a in apertures]
    
    #TODO: Generalise the above so that we aren't hard coding area limit.
    # Change code so that the largest n areas are grabbed where n is number
    # of balls/aperturs.
        
    centroids = [(sub[1], sub[0]) for sub in centroids]
    
    return centroids

def get_ball_positions(image, binary, number_of_balls):
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
                                            total_num_peaks=number_of_balls)
    ball_positions = list(zip(cx,cy))
    
    # order balls by y position.
    #TODO - we will have to generalise this for geometry agnostic behaviour
    ball_positions = sorted(ball_positions, key=itemgetter(1)) 
    
    return ball_positions



def sparse_image(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


#progress bar
def update_progress(progress):
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
    barLength = 100 # Modify this to change the length of the progress bar
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
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
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
    fig.suptitle('PoderBallz: A tale of four balls')
    
    #TODO: Generalise in for loop for any number of balls:
    
    df.plot(kind="scatter", x='Gantry', y=(what, 1, 'x'), s=3,
            color='darkred', label='b1', ax=ax1)
    df.plot(kind="scatter", x='Gantry', y=(what, 2, 'x'), s=3,
            color='darkblue', label='b2', ax=ax1)
    df.plot(kind="scatter", x='Gantry', y=(what, 3, 'x'), s=3,
            color='darkgreen', label='b3', ax=ax1)
    df.plot(kind="scatter", x='Gantry', y=(what, 4, 'x'), s=3,
            color='darkslategrey', label='b4', ax=ax1)
    
    df.plot(kind="scatter", x='Gantry', y=(what, 1, 'y'), s=3,
            color='darkred', label='b1', ax=ax2)
    df.plot(kind="scatter", x='Gantry', y=(what, 2, 'y'), s=3,
            color='darkblue', label='b2', ax=ax2)
    df.plot(kind="scatter", x='Gantry', y=(what, 3, 'y'), s=3,
            color='darkgreen', label='b3', ax=ax2)
    df.plot(kind="scatter", x='Gantry', y=(what, 4, 'y'), s=3,
            color='darkslategrey', label='b4', ax=ax2)
     
    ax1.set_ylabel('X position')  
    ax2.set_ylabel('Y position')
    
    plt.show()
    
    

def plot_coords_on_images(what):
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
    if what == 'drr':
        folder == drrfolder
        aperture_dflabel == 'DRRApertures'
        balls_dflabel == 'DRRBalls'
    elif what == 'epid':
        folder = epidfolder
        aperture_dflabel == 'EPIDApertures'
        balls_dflabel == 'EPIDBalls'
        
    
    
    for i in range(50,71):
        # test plotting to see if the coordinates makes sense
        filename = df.loc[i, 'filename'].values[0]
        filename = folder / filename
        im = imageio.imread(filename)
        im = np.array(im)
        
        if what == 'epid':
            balls = df.loc[i, balls_dflabel].values
            apertures = df.loc[i, aperture_dflabel].values
        elif what == 'drr':
            balls = df.loc[i, balls_dflabel].values
            apertures = df.loc[i, aperture_dflabel].values 
        #convert to format required by plotting function
        it = iter(balls)
        balls = [*zip(it,it)]
        
        it = iter(apertures)
        apertures = [*zip(it,it)]
        image = sparse_image(image, 900, 900)
        
        
        fig, ax = plt.subplots()
        image = color.gray2rgb(image)
    
        
        for a in apertures:
                ax.plot(a[0], a[1], color="darkred", marker='x', 
                        linewidth=3, markersize=5)
        
        for b in balls:
                ax.plot(b[0], b[1], color="darkblue",marker='o', 
                        linewidth=3, markersize=2)
        
        ax.imshow(image)
        plt.show()
    
def get_epid_balls_and_apertures(names, number_of_balls
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
    
        apeture_centroids = get_apeture_centroids(sel, binary, number_of_balls)
        apeture_centroids = [item for t in apeture_centroids for item in t]
        apeture_centroids = [int(item) for item in apeture_centroids]
   
        ball_positions = get_ball_positions(sel, binary, number_of_balls)
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
        
        # Progress bar
        update_progress(i/progmax)


def get_drr_balls(names):
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
    
        ball_positions = get_ball_positions(sel, binary)
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
        update_progress(i/progmax)

def get_drr_apertures(names):
    """
    Gets apeture positions from the mlc image.

    Parameters
    ----------
    files : list of strings
        contains the names of tif image files.

    Returns
    -------
    None.

    """

    for i, n in enumerate(names):
        filename = mlcfolder / n
        im = imageio.imread(filename)
        im = np.array(im)
        
        im = sparse_image(im, cropx, cropy)
        thresh = threshold_otsu(im)
        binary = im > thresh
        sel = np.zeros_like(im)
        sel[binary] = im[binary]
    
        aperture_centroids = get_apeture_centroids(sel, binary)
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
        
        # Progress bar
        update_progress(i/progmax)



data_folder = ('C:/Users/kanke/OneDrive/Work Miscellenous/SRS Geometric '+
                'Accuracy/PoderBallz/Output Images')
data_folder = Path(data_folder)
frameinfo = data_folder / 'Gantry_Angles.csv'
epidfolder = data_folder / 'EPID'
drrfolder = data_folder / 'DRR'
mlcfolder = data_folder / 'MLC'

#create dataframe with gantry angles and filenames

item_type = ["EPIDBalls","EPIDApertures","DRRBalls","DRRApertures"]
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
names = [os.path.basename(x) for x in glob.glob('C:/Users/kanke/OneDrive/'+
                                                'Work Miscellenous/SRS'+
                                                ' Geometric Accuracy/'+
                                                'PoderBallz/Output '+
                                                'Images/EPID/*.tif')]
df['filename'] = names



progmax = len(df)

#Process all images and save ball and aperture positions.
cropx = 900
cropy = 900


# get_epid_balls_and_apertures(names)

# get_drr_balls(names)

get_drr_apertures(names)

# plot_coords_on_images('drr') # Allowed arguments:'drr', 'epid'
    

# plot_against_gantry('EDIDBalls') # EPIDBalls, EPIDApertures, DRRBalls, DRRAperturs



# Scratch




























