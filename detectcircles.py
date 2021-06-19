# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:19:56 2021

@author: 56153805
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import find_contours, regionprops, label
from skimage.filters import threshold_otsu
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy.ndimage.measurements import center_of_mass



def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]



# Load picture and detect edges
image = imageio.imread('P:/14 Projects/49_SRS Phantom/Output Images/EPID/'+
                       'Image0012.tif')
image = np.array(image)

# Test images
#"P:\14 Projects\49_SRS Phantom\Ballz,Poder_6XFFF_210505_1627\MV\Ch0_1_668_173.24.tiff"

image = crop_center(image, 900, 900)
thresh = threshold_otsu(image)
binary = image > thresh
sel = np.zeros_like(image)
sel[binary] = image[binary]

def get_apeture_centroids(image):
    label_image = label(binary)
    apertures = regionprops(label_image)
    centroids = [a.centroid for a in apertures]
    
    # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 3, 1)
    # ax[1] = plt.subplot(1, 3, 2)
    # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
    
    # ax[0].imshow(image, cmap=plt.cm.)
    # ax[0].set_title('Original')
    # ax[0].axis('off')
    
    # ax[1].hist(image.ravel(), bins=256)
    # ax[1].set_title('Histogram')
    # ax[1].axvline(thresh, color='r')
    
    # ax[2].imshow(binary, cmap=plt.cm.gray)
    # ax[2].set_title('Thresholded')
    # ax[2].axis('off')
    
    # plt.show()
    
    #Find contours at a constant value of 0.8
    contours = find_contours(binary, 0.8)


    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(label_image, cmap=plt.cm.Blues)

    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    for centroid in centroids:
        ax.plot(centroid[1], centroid[0], marker='o', linewidth=2, markersize=2)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    return contours, centroids

contours, centroids = get_apeture_centroids(image)

edges = canny(image, sigma=3, low_threshold=5, high_threshold=10, mask=binary)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
ax.imshow(edges, cmap='gray')
ax.set_title('lowT=0, highT=50')

# Detect two radii
hough_radii = np.arange(4, 15)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 4 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=4)

# Draw them
fig, ax = plt.subplots()
image = color.gray2rgb(image)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius,
#                                     shape=image.shape)
#     image[circy, circx] = (500, 20, 20)

for centroid in centroids:
        ax.plot(centroid[1], centroid[0], color="darkred", marker='x', 
                linewidth=3, markersize=5)

for center_y, center_x in zip(cy, cx):
        ax.plot(center_x, center_y, color="darkblue",marker='o', 
                linewidth=3, markersize=2)

ax.imshow(image)
plt.show()

