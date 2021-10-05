# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:20:28 2021

Plotting tools for PoderBallz

@author: 56153805
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas
import imageio
import skimage.color as color


def sparse_image(img, cropx, cropy):
    """
    Return cropped image. Future functionality to make image sparse.
    
    Parameters
    ----------
    img : numpy array
        image to make sparse.
    cropx : TYPE
        crop dimension in x direction.
    cropy : TYPE
        crop dimension in y direction.

    Returns
    -------
    TYPE
        numpy array.

    """
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


# Plotting the balls
def plot_against_gantry(what, num_of_balls, df):
    """
    Plot ball positions over gantry angles.

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
    
def plot_coords_on_images(what, range_of_images, data_folder, df):
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
    epidfolder = data_folder / 'EPID'
    drrfolder = data_folder / 'DRR'
    mlcfolder = data_folder / 'MLC'
    
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
  
            

def boxplot(what, num_of_balls, df):
    
    # # Creating dataset
    # np.random.seed(10)
    
    # static HTT results from commissioning:
    
    # data_4 = [0.80,	0.20, 0.00,	0.90, 1.20,	0.20, 0.10,	0.70, 0.40,	0.10, 0.80,	0.70]
    # data_2 = [0.50, 0.20, 0.00, 0.70, 1.10, 0.10, 0.20,	0.10, 0.30,	0.20, 0.50,	0.30]
    # data_3 = [0.20,	0.40, 0.20,	0.50, 1.20,	0.10, 0.20,	0.40, 0.60,	0.30, 0.70,	0.30]
    # data_1 = [0.00,	0.20, 0.10,	0.20, 0.70,	0.20, 0.10,	0.30, 0.30,	0.70, 0.20,	0.00]
    
    
    # data_1 = np.random.normal(0.2, 0.1, 10)
    # data_2 = np.random.normal(0.32, 0.12, 10)
    # data_3 = np.random.normal(0.31, 0.15, 10)
    # data_4 = np.random.normal(0.6, 0.24, 10)
    
    
    data_1 = (df['WL', 1,'x']**2*df['WL', 1,'y']**2)**(1/2)
    data_2 = (df['WL', 2,'x']**2*df['WL', 2,'y']**2)**(1/2)
    data_3 = (df['WL', 3,'x']**2*df['WL', 3,'y']**2)**(1/2)
    data_4 = (df['WL', 4,'x']**2*df['WL', 4,'y']**2)**(1/2) 
    
    data = [data_1, data_2, data_3, data_4]
    
    fig = plt.figure(figsize =(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
     
    # Creating plot
    bp = ax.boxplot(data)
    ticks = plt.xticks([1,2,3,4],[30,50,60,100])
    plt.title(label="MTSI Positional Verification", fontsize=25)
    plt.xlabel("Distance from Isocentre (mm)", fontsize=10)
    plt.ylabel("Deviation from DRR (mm)", fontsize=10)
    
      
## removing outliers
def remove_outliers(df, what):
    
    #what = 'EPIDBalls'
    
    
    window = 10
    num_stds = 3
    
    
    x = df[what, 1, 'x'].copy()
    
    # add values to front for rolling average to work at start.
    m = np.pad(x, pad_width=(window-1, 0), mode='wrap')
    m = pandas.Series(m)
    m = m.rolling(window).median()
    
    # crop the NAN values
    m = m.iloc[window-1:]
    m.reset_index(drop=True, inplace=True)
    
    s = np.pad(m, pad_width=(window-1, 0), mode='wrap')
    s = pandas.Series(s)
    s = s.rolling(window).std()
    
    s = s.iloc[window-1:]
    s.reset_index(drop=True, inplace=True)
    
    xbool = (x <= m+num_stds*s) & (x >= m-num_stds*s)
    
    x2 = x.copy()
    x2 = x2.mask(~xbool)


    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('window = {swin}, stdevs = {snumstds}'.format(swin = window, snumstds = num_stds))
    x2.plot(ax=ax1)
    plt.show()



