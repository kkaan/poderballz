# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:02:40 2020

inputs: x, y, z room coordinates of ball with isocentre placed at the origin
        also margin m around which to shape the MLCs.

Transformation of ballshot location when Ganry rotates:

x' =  x cos(G) + y sin(G)
y' - -x sin(G) + y cos(G)
z' = z

Projection of postions to the isocentre plane:
    X = x' * 100/(100-y')
    Z = z' * 100/(100-y')
    

Calcultion for selecting which leaves to open up

    
@author: Kaan
"""

def makeaperture(x,y,z,margin):
    
    