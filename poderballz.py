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


def get_apeture_centroids(image, binary):
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
    centroids = [a.centroid for a in apertures]
    centroids = [(sub[1], sub[0]) for sub in centroids]
    
    return centroids

def get_ball_positions(image, binary):
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
                                            total_num_peaks=4)
    ball_positions = list(zip(cx,cy))
    
    # order balls by y position.
    # - we will have to generalise this for geometry agnostic behaviour
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
def plot_balls():
    """
    
    Plots ball positions over gantry agnles
    

    Returns
    -------
    None.

    """

    # Getting the ball positions in clear plotable structure from the dataframe
    g = list(image_df.Gantry.values)
    a = list(image_df.EPIDBalls.values)
    
    b1x = [i[0][0] for i in a]
    b1y = [i[0][1] for i in a]
    
    b2x = [i[1][0] for i in a]
    b2y = [i[1][1] for i in a]
    
    b3x = [i[2][0] for i in a]
    b3y = [i[2][1] for i in a]
    
    b4x = [i[3][0] for i in a]
    b4y = [i[3][1] for i in a]
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('PoderBallz: A tale of four balls')
    
    ax1.plot(g, b1x, 'ro', label='B1', markersize=1)
    ax1.plot(g, b2x, 'bo', label = 'B2', markersize=1) 
    ax1.plot(g, b3x, 'go', label = 'B3', markersize=1)
    ax1.plot(g, b4x, 'mo', label = 'B4', markersize=1)
    ax1.set_ylabel('X position of the balls')
    
    ax2.plot(g, b1y, 'ro', label='B1', markersize=1)
    ax2.plot(g, b2y, 'bo', label = 'B2', markersize=1) 
    ax2.plot(g, b3y, 'go', label = 'B3', markersize=1)
    ax2.plot(g, b4y, 'mo', label = 'B4', markersize=1)
    ax2.set_xlabel('Gantry')
    ax2.set_ylabel('Y position of the balls')
    
    
    plt.show()

def plot_coords_on_images(image, apertures, balls):
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
    
#def get_ballandapeture(files, balls_col_name, apetures_col_name):
    

#load folder
data_folder = 'P:/14 Projects/49_SRS Phantom/Output Images/'
data_folder = Path(data_folder)
frameinfo = data_folder / 'Gantry_Angles.csv'
epidfolder = data_folder / 'EPID'
drrfolder = data_folder / 'DRR'
mlcfolder = data_folder / 'MLC'

#create dataframe with gantry angles and filenames

item_type = ["EPIDBalls","EPIDApetures","DRRBalls","DRRApetures"]
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
names = [os.path.basename(x) for x in glob.glob('P:/14 Projects/49_SRS'+
                                                ' Phantom/Output'+
                                                ' Images/EPID/*.tif')]
df['filename'] = names

#get_ballandapeture(files, "EPIDBalls", "EPIDApertures")

progmax = len(df)

#Process all images and save ball and aperture positions.
cropx = 900
cropy = 900

for i, n in enumerate(names):
    filename = epidfolder / n
    im = imageio.imread(filename)
    im = np.array(im)
    
    im = sparse_image(im, cropx, cropy)
    thresh = threshold_otsu(im)
    binary = im > thresh
    sel = np.zeros_like(im)
    sel[binary] = im[binary]
    apeture_centroids = get_apeture_centroids(sel, binary)
    apeture_centroids = [item for t in apeture_centroids for item in t]
    apeture_centroids = [int(item) for item in apeture_centroids]
    
    ball_positions = get_ball_positions(sel, binary)
    ball_positions = [item for t in ball_positions for item in t]
    #the following needs to be changed to fit the new dataframe format:
    
    try:
        df.at[i, 'EPIDApetures'] = apeture_centroids
        df.at[i, 'EPIDBalls'] = ball_positions
    except AssertionError as error:
        print(error)
        print('Probably found too many balls or apertures."
              + "Change detection settings')
    
    # df.at[] is faster than df.loc[] but will preserve data type of df series. 
    # and it will do it silently. Saving floats in int columns will be lossful
    
    # Progress bar
    update_progress(i/progmax)


# for i in df.loc[50:70].itertuples():
#     filename = i.filename
#     filename = epidfolder / filename
#     im = imageio.imread(filename)
#     im = np.array(im)
    
#     balls = i.EPIDBalls
#     aperture = i.EPIDApertures
#     plot_coords_on_images(im, aperture, balls)
    
# plot_balls()



# Scratch




























