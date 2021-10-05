# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:09:59 2021

@author: 56153805
"""

import sys

#progress bar
def update_progress(progress, subtext):
    '''
    Creates a progress bar in standard output.

    Parameters
    ----------
    progress : int, float
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    barLength = 20 # Modify this to change the length of the progress bar
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
    text = "\rPercent: [{0}] {1}% {2} {3}".format( "#"*block + 
                                                  "-"*(barLength-block),
                                                  int(progress*100),
                                                  status,
                                                  subtext)
    sys.stdout.write(text)
    sys.stdout.flush()