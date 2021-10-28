# -*- coding: utf-8 -*-
"""Module contains detection algorithms for use in multiprocesses."""

import imageio
import numpy as np
from operator import itemgetter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
import cv2
from skimage.measure import regionprops, label


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

def get_ball_positions(image, binary, num_of_balls):
    """Return the positions of all the balls found in the image.
    
    This function is used by the funtions get_drr_balls and get_epid_balls.

    Parameters
    ----------
    image : numy_array of binarised imaged
        DRR image with balls clearly visible.
    binary: numpy_array
        Masked DRR for thresholded regions to minimise proecssing time.
    num_of_balls: integer
        Number of balls expected.

    Returns
    -------
    ball_positions : list
        alll ball positions in the image returned.

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


    #############################################################
    #  CV2 - uncommented this next block and comment all above  #
    #############################################################
        
    # blur = cv2.GaussianBlur(image, (9,9), 0)
    
    # # threshold
    # thresh = cv2.threshold(blur,128,255,cv2.THRESH_BINARY)[1]
    
    
    # # apply close and open morphology to smooth
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # # draw contours and get centroids
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # cx = []
    # cy = []
    # for cntr in contours:
    #     M = cv2.moments(cntr)
    #     cx.append(int(M["m10"] / M["m00"]))
    #     cy.append(int(M["m01"] / M["m00"]))
    # ball_positions = list(zip(cx,cy))
    
    # # order balls by y position.
  
    # ball_positions = sorted(ball_positions, key=itemgetter(1)) 
    
    # #TODO - we will have to generalise this for geometry agnostic behaviour
    # return ball_positions


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

def get_epid_balls_and_apertures_pool(name, i, num_of_balls, folder, cropx, cropy):
    """Get the ball and apeture positions from the EPID image.
    
    Uses skimage hough transform to grab apetures and ball centroid from epid
    images.

    Parameters
    ----------
    name : string
        name of image.
    i : int
        DESCRIPTION.
    num_of_balls : TYPE
        DESCRIPTION.
    folder : TYPE
        DESCRIPTION.
    cropx : TYPE
        DESCRIPTION.
    cropy : TYPE
        DESCRIPTION.

    Returns
    -------
    ball_positions : TYPE
        DESCRIPTION.
    apeture_centroids : TYPE
        DESCRIPTION.

    """
    filename = folder / name
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
    
    return ball_positions, apeture_centroids
    
      
def get_drr_balls_pool(name, i, num_of_balls, drrfolder, cropx, cropy):
    """
    Get the ball positions from the drr image.

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    num_of_balls : int
        DESCRIPTION.
    drrfolder : windowspath object
        DESCRIPTION.
    cropx : integer
        DESCRIPTION.
    cropy : integer
        DESCRIPTION.

    Returns
    -------
    ball_positions : list
        DESCRIPTION.

    """
    filename = drrfolder / name
    
    
    im = imageio.imread(filename)
    im = np.array(im)
    img = sparse_image(im, cropx, cropy)
    

    blur = cv2.GaussianBlur(img, (9,9), 0)
    
    # threshold
    thresh = cv2.threshold(blur,128,255,cv2.THRESH_BINARY)[1]
    
    
    # apply close and open morphology to smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # draw contours and get centroids
    circles = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cx = []
    cy = []
    for cntr in contours:
        cv2.drawContours(circles, [cntr], -1, (0,0,255), 2)
        M = cv2.moments(cntr)
        cx.append(int(M["m10"] / M["m00"]))
        cy.append(int(M["m01"] / M["m00"]))

    # reorder balls by y position. This determines ball number
    ball_positions = list(zip(cx,cy))
    ball_positions = sorted(ball_positions, key=itemgetter(1)) 
    
    # orientate for insertion into dataframe
    ball_positions = [item for t in ball_positions for item in t]
    
    return ball_positions

    # try:
    #     df.at[i, 'DRRBalls'] = ball_positions
    #         # df.at[] is faster than df.loc[] but will preserve data 
    #         # type of df series. And it will do it silently. Saving 
    #         # floats in int columns will be lossful.
    # except AssertionError as error:
    #     print(error)
    #     print("Probably found too many balls or apertures." + 
    #           "Change detection settings")
        
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
    # text = "Finding balls in DRR {0}".format(n)
   
    # update_progress(i/progmax, text)

def get_drr_apertures_pool(name, num_of_balls, mlcfolder, cropx, cropy):
    """ 
    get_drr_apertures_pool(name, mlcfolder, num_of_balls, cropx, cropy)
    
    MLC files in tiff format are analysed to obtain centre of apoertures
    

    Parameters
    ----------
    name : string
        name of file.
    mlcfolder : string
        folder of files.
    num_of_balls : int
        how many balls.
    cropx : int
        crop dimensions.
    cropy : int
        crop dimensions.

    Returns
    -------
    aperture_centroids : list
        returns ball position x,y in order of y axis location.

    """
    
    filename = mlcfolder / name
    
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
    return aperture_centroids
    
# def main():
#     """
#     Suppress execution on import.
    
#     Returns
#     -------
#     None.

#     """
#     pass

# if __name__ == "__main__":
#    # stuff only to run when not called via 'import' here
#    main()