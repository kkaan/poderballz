# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:19:56 2021

@author: 56153805
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from PIL import Image

# Load picture and detect edges
image = Image.open('P:/14 Projects/49_SRS Phantom/Ballz,Poder_6XFFF_210505_1627/MV/Ch0_1_993_45.47.tiff')
image = np.array(image)
edges = canny(image, sigma=3, low_threshold=700, high_threshold=670)


# Detect two radii
hough_radii = np.arange(20, 35, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=4)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()