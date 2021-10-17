# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:01:12 2021

@author: Joshua Banks Mailman, Ph.D.
"""

import numpy as np 
import matplotlib.pyplot as plt


def graph(x_array, y_array, custom_title, scatter=True, solid_line=False, line_viz = None):
    ''' This function can accept basic x and y arrays for a basic 
    scatter plot or line plot (or both), but it can also accept, 
    in its line_viz parameter, a list of x, y array pairs.
    Its configuration for coloring, line weight, and dot size are 
    tailored specifically for the smoothing examples presented in 
    this blog.'''
    
    # Set up the plotting
    figure_proportions = (10, 5)
    plt.figure(figsize=figure_proportions )    
    x_min, x_max = -3.3, 4.3
    y_max = 100
    y_axis_label = 'altitude (feet)'
    x_axis_label = 'time (seconds from arrival at target)'
    title = 'Trajectory of flight'
    custom_title = title + ' - ' + custom_title
    plt.title(custom_title)         
    dot_opacity = line_opacity = 1
    size = 30 if len(x_array) > 15 else 100 
    line_weight=1
    
    # Handle the cases where lines or curves are visualized, 
    # beyond the obvious
    if line_viz != None:
        if len(line_viz) == 1:
            colors = ['green']
        if len(line_viz) ==2:
            colors = ['lightseagreen', 'olive']
            dot_opacity = .3
            line_weight = 3
        if len(line_viz) == 4:
            dot_opacity = .1
            line_opacity = .1
            colors = ['turquoise', 'magenta','tomato', 'gold']           
        for i, (x, y) in enumerate(line_viz):
            plt.plot(x, y, color = colors[i], lw=3)
    
    # Handle the standard cases
    if scatter:
        plt.scatter(x_array, y_array, color = 'green', marker = "o", alpha = dot_opacity, s=size) 
    if solid_line:
        plt.plot(x_array, y_array, color = 'green', alpha = line_opacity)
    # Present the plotting
    plt.xlabel(x_axis_label) 
    plt.ylabel(y_axis_label) 
    plt.xlim(x_min, x_max )
    plt.ylim(0, y_max )
    plt.savefig('viz/'+custom_title+'.svg')
    plt.show()
    
def main():
    """
    Suppress execution on import.
    
    Returns
    -------
    None.

    """
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()