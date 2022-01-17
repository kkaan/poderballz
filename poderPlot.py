# -*- coding: utf-8 -*-
"""
Plotting tools for PoderBallz.

Created on Fri Jun  4 11:20:28 2021

.. module:: poderballz
   :platform: Windows
   :synopsis: grab some balls.

.. moduleauthor:: Kaan Kankean <kankean.kandasamy@health.nsw.gov.au>

"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
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

def plot_a_ball(ballnumber, df):
    """
    Plot the position of ball and corresponding apertures.

    Parameters
    ----------
    ball_number : TYPE
        which ball to plot.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    item_type = ['EPIDBalls', 'EPIDApertures', 'DRRBalls', 'DRRApertures']
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('PoderBallz A tale of a ball {}'.format(ballnumber))
    
    colours = ['darkred', 'darkblue', 'darkgreen', 'darkslategrey']
    # need more colours for more balls.
    
    # num_of_balls = 1 # comment out this line to print all balls
    
    for i in range(0,4): # epid ball, epid apertures, drr balls and drr apertures
        df.plot(kind="scatter", x='Gantry', y=(item_type[i], ballnumber, 'x'), s=3, alpha =0.3,
                color=colours[i-1], label=' {}'.format(item_type[i]), ax=ax1)
          
    ax1.set_ylabel('X position (mm)')  
    
    
    ax1.legend(title="None", fontsize="xx-small", loc="upper right")
    plt.show()
    plt.close('all')


# Plotting the balls
def plot_against_gantry(what, num_of_balls, df):
    """
    Plot ball positions over gantry angles.

    Returns 
    -------
    None.

    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Postions of four {} over gantry rotation'.format(what))
    
    colours = ['darkred', 'darkblue', 'darkgreen', 'darkslategrey']
    # need more colours for more balls.
    
    # num_of_balls = 1 # comment out this line to print all balls
    
    for i in range(1,num_of_balls+1):
        df.plot(kind="scatter", x='Gantry', y=(what, i, 'x'), s=10,
                color=colours[i-1], label='ball {}'.format(i), ax=ax1)
        
        df.plot(kind="scatter", x='Gantry', y=(what, i, 'y'), s=10,
                color=colours[i-1], label='ball {}'.format(i), ax=ax2)
    
    ax1.set_ylabel('X position (mm)')  
    ax2.set_ylabel('Y position (mm)')
    
    ax1.legend(title="Target #", fontsize="small", loc="upper right")
    ax2.get_legend().remove()
    
def plot_coords_on_images(what, range_of_images, data_folder, df):
    """
    Plot the coordinates of balls and apertures on the images.
    
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

    elif what == 'drr':
        folder = mlcfolder
        aperture_dflabel = 'DRRApertures'
        balls_dflabel = 'DRRBalls'

    else:
        print('need to choose: drrballs, epid, drrape')
        

        
    
    
    for i in range_of_images:
        # test plotting to see if the coordinates makes sense
        filename = df.loc[i, 'filename'].values[0]
        filename = folder / filename
        im = imageio.imread(filename)
        im = np.array(im)
        
        if what == 'epid':
            balls = df.loc[i, balls_dflabel].values
            apertures = df.loc[i, aperture_dflabel].values
        elif what == 'drrballs' or 'drrape':
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
    
        if what == 'drrape':
            for a in apertures:
                    ax.plot(a[0], a[1], color="darkred", marker='x', 
                            linewidth=5, markersize=10)
        elif what == 'drrballs':
            for b in balls:
                    ax.plot(b[0], b[1], color="darkblue",marker='o', 
                            linewidth=3, markersize=6)
        elif what == 'epid':
            for a in apertures:
                    ax.plot(a[0], a[1], color="darkred", marker='x', 
                            linewidth=5, markersize=10)
            for b in balls:
                    ax.plot(b[0], b[1], color="darkblue",marker='o', 
                            linewidth=3, markersize=6)
                    
        ax.grid(True)            
        ax.imshow(image)
        #ax.title.set_text(filename)
  
            

def boxplot(what, num_of_balls, df):
    """
    Display boxplot of WL results for all the targets.

    Parameters
    ----------
    what : TYPE
        DESCRIPTION.
    num_of_balls : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # # Creating dataset
    # np.random.seed(10)
    
    # static HTT results from commissioning:
    
    # data_4 = [0.80,	0.20, 0.00,	0.90, 1.20,	0.20, 0.10,	0.70, 0.40,	0.10, 0.80,	0.70]
    # data_2 = [0.50, 0.20, 0.00, 0.70, 1.10, 0.10, 0.20,	0.10, 0.30,	0.20, 0.50,	0.30]
    # data_3 = [0.20,	0.40, 0.20,	0.50, 1.20,	0.10, 0.20,	0.40, 0.60,	0.30, 0.70,	0.30]
    # data_1 = [0.00,	0.20, 0.10,	0.20, 0.70,	0.20, 0.10,	0.30, 0.30,	0.70, 0.20,	0.00]
    
    
    # static
    # data_1 = np.random.normal(0.2, 0.1, 10)
    # data_2 = np.random.normal(0.32, 0.12, 10)
    # data_3 = np.random.normal(0.31, 0.15, 10)
    # data_4 = np.random.normal(0.6, 0.24, 10)
    
    
    ball_1 = (df['WL', 1,'x']**2*df['WL', 1,'y']**2)**(1/2)
    ball_2 = (df['WL', 2,'x']**2*df['WL', 2,'y']**2)**(1/2)
    ball_3 = (df['WL', 3,'x']**2*df['WL', 3,'y']**2)**(1/2)
    ball_4 = (df['WL', 4,'x']**2*df['WL', 4,'y']**2)**(1/2) 
    
    # in order of distance from isocentre
    data = [ball_3, ball_4, ball_2, ball_1]
    
    fig = plt.figure(figsize =(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
     
    # Creating plot
    bp = ax.boxplot(data)
    ticks = plt.xticks([1,2,3,4],[0,60,70,100])
    plt.title(label="MTSI Positional Verification", fontsize=25)
    plt.xlabel("Distance from Isocentre (mm)", fontsize=10)
    plt.ylabel("Deviation from DRR (mm)", fontsize=10)
    
      
## removing outliers




