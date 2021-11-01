# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:37:21 2021

@author: 56153805
"""
import pylab
import dicom

ImageFile=dicom.read_file("P:\14 Projects\49_SRS Phantom\HTT Shifts\G180B_C0T0,PoderBallz_0.6mm AGU_6XFFF_210926_2317\MV\dcm") #Path to "*.dcm" file
#pylab.imshow(ImageFile.pixel_array,cmap=pylab.cm.bone) #to view image
pylab.imsave("P:\14 Projects\49_SRS Phantom\HTT Shifts\G180B_C0T0,PoderBallz_0.6mm AGU_6XFFF_210926_2317\MV\convertedtiff",ImageFile.pixel_array,cmap=pylab.cm.bone)

