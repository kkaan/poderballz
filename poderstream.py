# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 21:22:30 2022

@author: 56153805
"""

import streamlit as st
import stqdm

from pathlib import Path
import pandas


import glob
import os


import time
from itertools import repeat

import poderballs
from poderPlot import plot_against_gantry, boxplot, plot_coords_on_images, plot_a_ball


if __name__ == "__main__": 
    

    st.title("Poderballs")
    
    fstring = '//nsccgosfs04/physics1$/14 Projects/49_SRS Phantom/HTT/shift/HTT,0 shift_6FFF_211018_0747/Output Images'
    
    st.write(fstring)
    
    
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
    

    
       
    #load epid image names
    
    names = [os.path.basename(x) for x in glob.glob(fstring+'/EPID/*.tif')]
    df['filename'] = names
    
    progmax = len(df)-1
    
    cropx = 900
    cropy = 900
    
    # start = time.time()       
    # get_epid_balls_and_apertures(names, epidfolder, num_of_balls, cropx, 
    #                               cropy, df)
    # end = time.time()
    # print(' ')
    # print(end - start, 'seconds')
    
    
    # multiproccess version for getting epid balls and apetures
    # TODO: Tidy this up with a wrapper pool function.
    # TODO: Add error handling to exit out of pools on fails.
    # TODO: Ensure all processes are being closed
    start = time.time()
    nameindex = list(range(0,len(names)))
    
    
    # multiproccess version for getting epid balls and apertures
    inputs = zip(names, nameindex, repeat(num_of_balls), repeat(epidfolder), 
                  repeat(cropx), 
                  repeat(cropy))
    
    poderballs.epidpool(inputs, names, df)
    
    st.write(df.EPIDBalls)
    
    # multiproccess version for getting drr balls    
    inputs = zip(names, nameindex, repeat(num_of_balls), repeat(drrfolder), 
                  repeat(cropx), 
                  repeat(cropy))
    
    
    poderballs.drrpool(inputs, names, df)
    st.write(df.EPIDBalls)
    
    # multiproccess version for getting mlc aperture centroids    
    inputs = zip(names, repeat(num_of_balls), repeat(mlcfolder),
                  repeat(cropx), 
                  repeat(cropy))
    
    poderballs.mlcpool(inputs, names, df)
    
    st.write(df.EPIDBalls)
        
    fig1 = plot_against_gantry('DRRApertures', num_of_balls, df) # EPIDBalls, EPIDApertures, DRRBalls, DRRAperturs
    st.pyplot(fig1)
    
    
    # Allowed arguments:'drrballs', 'drrape', 'drr', 'epid' 
    # plot_coords_on_images('epid', range(28,53), data_folder, df)
    # plot_coords_on_images('drrape', range(28,53), data_folder, df)
    # plot_coords_on_images('drrballs', range(105,116), data_folder, df)
    
    # preserve a copy of observed values before interpolation
    df_observed = df.copy()
    
    # interpolate so we don't have pixel res artefacts
    poderballs.interpolate(df) # convert to mm and then interpolate.
    
    # if converted to mm then convert back to pixel values
    # to see the coordinates plotted on the images.
    dfpixel = df.copy()
    SDD = 500 #mm from iso
    pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    pixel_to_mm = pixel_res*(1000+SDD)/1000
    dfpixel.loc[:,item_type] = dfpixel.loc[:,item_type]/pixel_to_mm
    image1 = plot_coords_on_images('epid', range(0,100), data_folder, dfpixel)
    image2 = plot_coords_on_images('epid', range(0,100), data_folder, df_observed)    
    
    st.pyplot(image1)
    st.pyplot(image2)
    
    poderballs.calculateWL(df)
           
    
    fig2 = plot_against_gantry('WL', num_of_balls, df)
    st.pyplot(fig2)
    fig3 = boxplot('WL', num_of_balls, df)
    st.pyplot(fig3)