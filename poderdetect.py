# -*- coding: utf-8 -*-
"""Hidden Target and aperture detection"""
import sys
import cv2
import imageio
import numpy as np
from operator import itemgetter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from poderprogressbar import update_progress

def sparse_image(img,cropx,cropy):
    """Crop the imge and make it sparse.
    
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
    """Calculate the locations of all the mlc defined aperture positions.
    
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
    """Return the positions of all the balls found in the image.
    
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

def get_epid_balls_and_apertures(names, epidfolder, num_of_balls, 
                                 cropx, cropy, df):
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
        progmax = len(df.loc[:,'Gantry'])
        update_progress(i/progmax, text)
      
def get_drr_balls(names, drrfolder, num_of_balls, cropx, cropy, df):
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
        
        edges = canny(im, sigma=3, low_threshold=10, high_threshold=11, mask=binary)
        
        
        hough_radii = np.arange(4,20,3)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=num_of_balls)
        
        ball_positions = list(zip(cx,cy))
    
        # order balls by y position.
        
        ball_positions = sorted(ball_positions, key=itemgetter(1)) 
        
        # p
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
        
        # #debug lines
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        # img = color.gray2rgb(im)
        # for center_y, center_x, radius in zip(cy, cx, radii):
        #     circy, circx = circle_perimeter(center_y, center_x, radius,
        #                             shape=im.shape)
        #     img[circy, circx] = (120, 20, 1)
        #     ax.plot(center_x, center_y, color="darkred", marker='o', 
        #                     linewidth=3, markersize=3)
        # ax.imshow(img, cmap=plt.cm.RdBu)
        # plt.show()
        # print(hough_radii)
        # print('x1 y1 {} \nx2 y2 {} \nx3 y3 {} \nx4, y4 {}'.format(ball_positions[0:2],ball_positions[2:4],ball_positions[4:6],ball_positions[6:8]))
        # #end debug lines
        
        
        # Progress bar
        text = "Finding balls in DRR {0}".format(n)
        
        progmax = len(df.loc[:,'Gantry'])
        update_progress(i/progmax, text)
        


def get_drr_apertures(names, mlcfolder, num_of_balls, cropx, cropy, df):
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
        progmax = len(df.loc[:,'Gantry'])
        text = "Finding apertures in MLC {0}".format(n)
        update_progress(i/progmax, text)