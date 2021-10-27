# -*- coding: utf-8 -*-
"""
Multithread image processing
Created on Wed Oct 27 17:56:19 2021



@author: 56153805


"""

import imageio
import numpy as np
from operator import itemgetter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
import cv2



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

# def get_epid_balls_and_apertures_pool(names, num_of_balls):
#     """
#     Get the ball and apeture positions from the EPID image.

#     Parameters
#     ----------
#     files : list of strings
#         contains the names of tif image files.

#     Returns
#     -------
#     None.

#     """
#     for i, n in enumerate(names):
#         filename = epidfolder / n
#         im = imageio.imread(filename)
#         im = np.array(im)
        
#         im = sparse_image(im, cropx, cropy)
#         thresh = threshold_otsu(im)
#         binary = im > thresh
#         sel = np.zeros_like(im)
#         sel[binary] = im[binary]
    
#         apeture_centroids = get_apeture_centroids(sel, binary, num_of_balls)
#         apeture_centroids = [item for t in apeture_centroids for item in t]
#         apeture_centroids = [int(item) for item in apeture_centroids]
   
#         ball_positions = get_ball_positions(sel, binary, num_of_balls)
#         ball_positions = [item for t in ball_positions for item in t]
    
    
#         try:
#             df.at[i, 'EPIDApertures'] = apeture_centroids
#             df.at[i, 'EPIDBalls'] = ball_positions
#                 # df.at[] is faster than df.loc[] but will preserve data 
#                 # type of df series. And it will do it silently. Saving 
#                 # floats in int columns will be lossful.
#         except AssertionError as error:
#             print(error)
#             print("Probably found too many balls or apertures." +
#                   "Change detection settings")
        
#         #current status
#         text = "Finding balls and apertures in EPID {0}".format(n)
        
#         # Progress bar
#         update_progress(i/progmax, text)

def get_drr_balls_pool(name, i, num_of_balls, drrfolder, cropx, cropy):
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
    
    filename = drrfolder / name
    
    
    im = imageio.imread(filename)
    im = np.array(im)
    img = sparse_image(im, cropx, cropy)
    
    # CV option
    
    # read image
    #img = cv2.imread(filename)

    # convert to grayscale
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
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
        # x = round(cx)
        # y = round(cy)
        # #circles[y-2:y+3,x-2:x+3] = (0,255,0)
        # print(cx,",",cy)        
    
    # plt.imphow(thresh)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("circles", circles)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # save cropped image
    # cv2.imwrite('circles_thresh.png',thresh)
    # cv2.imwrite('circles_centroids.png',circles)
       
    # thresh = threshold_otsu(im)
    # binary = im > thresh
    
    # edges = canny(im, sigma=3, low_threshold=5, high_threshold=10, mask=binary)
    
    
    # hough_radii = np.arange(4,20,3)
    # hough_res = hough_circle(edges, hough_radii)
    # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
    #                                     total_num_peaks=num_of_balls)
    
    ball_positions = list(zip(cx,cy))

    # order balls by y position.
    
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