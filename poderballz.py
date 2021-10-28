# -*- coding: utf-8 -*-
"""
Calculate Winston Luts results for multiple targets in a Dynamic HTT.

Used alongside the a phantom with multiple ball bearing targets and an MLC 
based plan with DCAT like apertures centred on the targets. The analysis will 
provide details on the accuracy with which the planned apetures are delivered
on treatment for a VMAT field with dynamic MLC and gantry rotation.

.. module:: poderballz
   :platform: Windows
   :synopsis: grab some balls.

.. moduleauthor:: Kaan Kankean <kankean.kandasamy@health.nsw.gov.au>

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import glob
import os
import tqdm

from pathlib import Path
from multiprocessing import Pool
from itertools import repeat

from poderPlot import plot_against_gantry, boxplot, plot_coords_on_images, plot_a_ball
from poderInterpolate import interpolate
from poderpool import get_drr_balls_pool, get_epid_balls_and_apertures_pool

# equivalent sequential functions 
from poderdetect import get_drr_apertures, get_drr_balls, get_epid_balls_and_apertures

def calculateWL(df):
    """
    Calculate Winston Lutz deviations and saves to dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe of ball and aperture details.

    Returns
    -------
    None.

    """
    # calculate deviation in mm
    # Resolution details in DICOM header Image plane pixel spacing (3002,0011)
    # 0.336 mm/px at SID 100cm
    
    
    # SDD = 500 #mm from iso
    # pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    # pixel_to_mm = pixel_res*(1000+SDD)/1000
    
    pixel_to_mm = 1 # conversion is now done prior to interpolation.
       
    df['WL'] = pixel_to_mm*(df.loc[:, 'DRRBalls'] -df.loc[:, 'EPIDBalls'])-(
        pixel_to_mm*(df.loc[:, 'DRRApertures'] - df.loc[:, 'EPIDApertures']))
    
       
    # remove values higher than 5 mm 
    df['WL'] = df['WL'][np.abs(df.WL) < 5]

if __name__ == "__main__": 
    
    fstring = 'P:/14 Projects/49_SRS Phantom/HTT Shifts/HTT, 0 Shift 6FFF_2108_0747/Output Images'
    data_folder = (fstring)
    data_folder = Path(data_folder)
    frameinfo = data_folder / 'Gantry_Angles.csv'
    epidfolder = data_folder / 'EPID'
    drrfolder = data_folder / 'DRR'
    mlcfolder = data_folder / 'MLC'
    
    #create dataframe with gantry angles and filenames
    
    item_type = ["EPIDBalls","EPIDApertures","DRRBalls","DRRApertures", "WL"]
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
    
    names = [os.path.basename(x) for x in glob.glob(fstring+'/EPID/*.tif')]
    df['filename'] = names
    
    progmax = len(df)-1
    
    cropx = 900
    cropy = 900
    
    #Process all images and save ball and aperture positions.
    
    
    # get_epid_balls_and_apertures(names, epidfolder, num_of_balls, cropx, 
    #                              cropy, df)
    
    # multiproccess version for getting epid balls and apetures    
    nameindex = list(range(0,len(names)))
    inputs = zip(names, nameindex, repeat(num_of_balls), repeat(epidfolder), 
                  repeat(cropx), 
                  repeat(cropy))
    
    pool = Pool(15) 
    epidpoolballs, epidpoolapes = zip(*pool.starmap(get_epid_balls_and_apertures_pool, 
                                               tqdm.tqdm(inputs, 
                                                         total=len(names))))
    pool.close()
    pool.join()
    
    
    
    epidpoolballs = np.asarray(epidpoolballs)
    epidpoolapes = np.asarray(epidpoolapes)
    df.loc[:,'EPIDBalls']= epidpoolballs
    df.loc[:,'EPIDApertures']= epidpoolapes
   
      
    inputs = zip(names, nameindex, repeat(num_of_balls), repeat(drrfolder), 
                  repeat(cropx), 
                  repeat(cropy))
    
    pool = Pool(15) 
    drrpoolballs = pool.starmap(get_drr_balls_pool, 
                                tqdm.tqdm(inputs,
                                           total=len(names)))
    pool.close()
    pool.join()
    
    drrpoolballs = np.asarray(drrpoolballs)
    df.loc[:,'DRRBalls']= drrpoolballs
    #get_drr_balls(names, drrfolder, num_of_balls, cropx, cropy, df)
    get_drr_apertures(names, mlcfolder, num_of_balls, cropx, cropy, df)
    
    plot_against_gantry('DRRApertures', num_of_balls, df) # EPIDBalls, EPIDApertures, DRRBalls, DRRAperturs
    plt.show()
    plot_against_gantry('DRRBalls', num_of_balls, df)
    plt.show()
    plot_against_gantry('EPIDApertures', num_of_balls, df)
    plt.show()
    plot_against_gantry('EPIDBalls', num_of_balls, df)
    plt.show()
    
    # Allowed arguments:'drrballs', 'drrape', 'drr', 'epid' 
    plot_coords_on_images('epid', range(28,53), data_folder, df)
    plot_coords_on_images('drrape', range(28,53), data_folder, df)
    plot_coords_on_images('drrballs', range(105,116), data_folder, df)
    
    
    
    df_observed = df.copy()
    # copy of non interpolated values for diagnostics
    interpolate(df) # interpolate and convert to mm
    
    
    # if converted to mm then convert back to pixel values
    # to see the coordinates plotted on the images.
    dfpixel = df.copy()
    SDD = 500 #mm from iso
    pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    pixel_to_mm = pixel_res*(1000+SDD)/1000
    dfpixel.loc[:,item_type] = dfpixel.loc[:,item_type]/pixel_to_mm
    plot_coords_on_images('epid', range(105,116), data_folder, dfpixel)
    
    calculateWL(df)
           
    
    plot_against_gantry('WL', num_of_balls, df)
    plt.show()
    boxplot('WL', num_of_balls, df)
    plt.show()

    # plotting various things for debugging and comparison   
    plot_a_ball(2,df_observed)
    
    
    df.to_csv(fstring+'/results.csv')
    
    whattoplot = 'WL'
    whichball = 3
    legendtitle = whattoplot+' '+ str(whichball)
    ax1 = df.plot(kind="scatter", x="Gantry", y=(whattoplot,whichball,'x'), s=1, color='darkred', label='interpolated')
    df_observed.plot(kind='scatter', 
                      x="Gantry", y=(whattoplot,whichball,'x'), s=1, 
                      color='darkblue', label='cv_observed', ax=ax1)
    ax1.legend(title=legendtitle)
    plt.show()
    

    


















