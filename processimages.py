# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:19:56 2021

#detect balls and circles for all images in MV and tabulate with Gantry values
#detect balls and circles for all images in DRRR and tabulate wwith Gantry values
#do a polar plot
#calculate deviations for each gantry position.



@author: 56153805
"""
import time, sys
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import pandas 
from pathlib import Path

from operator import itemgetter
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import find_contours, regionprops, label
from skimage.filters import threshold_otsu
from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte
# from scipy.ndimage.measurements import center_of_mass
from PIL import Image



def get_apeture_centroids(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    
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
    #contours = find_contours(binary, 0.8)


    # Display the image and plot all contours found
    # fig, ax = plt.subplots()
    # ax.imshow(label_image, cmap=plt.cm.Blues)

    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    # for centroid in centroids:
    #     ax.plot(centroid[1], centroid[0], marker='o', linewidth=2, markersize=2)
    
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    
    return centroids

def get_ball_positions(sel):
    
    edges = canny(sel, sigma=2, low_threshold=10, high_threshold=50)
    hough_radii = np.arange(4, 15)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=4)
    ball_positions = list(zip(cy,cx))
    return ball_positions
    
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def import_DRRs():
    pass


#progress bar
def update_progress(progress):
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
    

#load folder
data_folder = Path("P:/14 Projects/49_SRS Phantom/Ballz,Poder_6XFFF_210505_1627/MV/")

frameinfo = data_folder / "Ch0_frame_info.csv"

image_df = pandas.read_csv(frameinfo, header=None, names=["Gantry", "Balls", "Apertures", "Whateva"])
image_df['Balls'] = image_df['Balls'].astype(object)
image_df['Apertures'] = image_df['Apertures'].astype(object)



# images = glob.glob("P:/14 Projects/49_SRS Phantom/Ballz,Poder_6XFFF_210505_1627/MV/*.tiff")
# for image in images:
#     with open(image, 'rb') as file:
#         img = Image.open(file)
        

# progress bar
progmax = len(image_df.index)
count = 0

for filename in image_df.index[40:]:
    
    image_file = data_folder / filename
    imgepid = Image.open(image_file)
    imgepid = np.array(imgepid)
    imgepid = crop_center(imgepid, 900, 900)
    thresh = threshold_otsu(imgepid)
    binary = imgepid > thresh
    sel = np.zeros_like(imgepid)
    sel[binary] = imgepid[binary]
    apeture_centroids = get_apeture_centroids(imgepid)
    ball_positions = get_ball_positions(sel)
    ball_positions = sorted(ball_positions, key=itemgetter(0))
    
    if len(apeture_centroids) == len(apeture_centroids) == 4:
        image_df.at[filename,'Apertures'] = apeture_centroids
        image_df.at[filename,'Balls'] = ball_positions
    
    # Progress bar
    count += 1
    update_progress(count/progmax)
 


#clean up the data 

#get rid of empty lists


#plot ball positions in relation to gantry

g = list(image_df.Gantry.values)
a = list(image_df.Balls.values)



b1x = [i[0][0] for i in a]
b1y = [i[0][1] for i in a]

b2x = [i[1][0] for i in a]
b2y = [i[1][1] for i in a]

b3x = [i[2][0] for i in a]
b3y = [i[2][1] for i in a]

b4x = [i[3][0] for i in a]
b4y = [i[3][1] for i in a]

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('A tale of four balls')

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
ax2.set_ylabel('X position of the balls')


plt.show()
   

# # Load DRR
# ds = dicom.dcmread('P:/14 Projects/49_SRS Phantom/DRRs/RI.PhysPLA.G180B_C0T0_9-NA.dcm')
# imgdrr = ds.pixel_array


# plt.imshow(imgdrr)

   

# edges = canny(sel, sigma=2, low_threshold=10, high_threshold=50)

# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
# # ax.imshow(edges, cmap='gray')
# # ax.set_title('lowT=0, highT=50')

# # Detect two radii
# hough_radii = np.arange(4, 15)
# hough_res = hough_circle(edges, hough_radii)

# # Select the most prominent 4 circles
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                             total_num_peaks=4)

# # Draw them
# fig, ax = plt.subplots()
# imgepid = color.gray2rgb(imgepid)
# # for center_y, center_x, radius in zip(cy, cx, radii):
# #     circy, circx = circle_perimeter(center_y, center_x, radius,
# #                                     shape=imgepid.shape)
# #     imgepid[circy, circx] = (500, 20, 20)

# for centroid in centroids:
#         ax.plot(centroid[1], centroid[0], color="darkred", marker='x', linewidth=3, markersize=5)

# for center_y, center_x in zip(cy, cx):
#         ax.plot(center_x, center_y, color="darkblue",marker='o', linewidth=3, markersize=2)

# ax.imshow(imgepid)
# plt.show()

