#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:56:15 2020

@author: ashley
"""
from array import *


file_rev    = "H"
treat_type  = "Dynamic Dose"
mlc_model   = "Varian HD120"
tolerance   = 0.1

cp_total    = 51

gap_width  = 4.0  #cm
left_edge  = 5  #cm
right_edge = 5  #cm

leaf_lower = 13 # Y=5
leaf_upper = 48 # Y=5

lfo_offset = 0 #mm
increment = (right_edge + left_edge + gap_width) / (cp_total - 1)


lastname   = "Gap_" + str(gap_width) + "cm"
firstname  = str(lfo_offset) + "mm RFO"
pat_id     = "HDMLC_DLG"


filename = str(gap_width) + "cmGAP_" + str(lfo_offset) + "mmRFO_" + str(cp_total) + "CP.dva"


# Write header information first
str_output = "File Rev = " + file_rev + "\n" \
    + "Treatment = " + treat_type + "\n" \
    + "Last Name = " + lastname + "\n" \
    + "First Name = " + firstname + "\n" \
    + "Patient ID = " + pat_id + "\n" \
    + "Number of Fields = " + str(cp_total) + "\n" \
    + "Model = " + mlc_model + "\n" \
    + "Tolerance = " + str(tolerance) + "\n"

A_init = [0.0] * 61
B_init = [0.0] * 61


A_pos = [0.0] * 61
B_pos = [0.0] * 61

for i in range(0, 61):
    if i < leaf_lower:
        A_init[i] = 0
        B_init[i] = 0
    elif i > leaf_upper:
        A_init[i] = 0
        B_init[i] = 0
    else:
        A_init[i] = -left_edge
        B_init[i] = left_edge + gap_width
        
    print("A" + str(i) + " = " + str(A_init[i]))
    print("B" + str(i) + " = " + str(B_init[i]))
    
for field in range(0,cp_total):
    str_output += "\nField = " + str(field) + "\n"
    
    index = field/(cp_total - 1)
    str_output += "Index = " + str(index) + "\n" \
        + "Carriage Group = 1\n" \
        + "Operator = \n" \
        + "Collimator = 0.0\n"
    
    for i in range(1, 61):
        if i < leaf_lower:
            A_pos[i] = 0
            B_pos[i] = 0
        elif i > leaf_upper:
            A_pos[i] = 0
            B_pos[i] = 0
        else:
            A_pos[i] = A_init[i] + increment * field - (lfo_offset / 10)
            B_pos[i] = B_init[i] - increment * field - (lfo_offset / 10)
    
    
    # A Bank
    for i in range(1, 61):
        if (i<10):
            str_output += "Leaf  " + str(i) + "A = " + format(A_pos[i], '.3f') + "\n"
        else:
            str_output += "Leaf " + str(i) + "A = " + format(A_pos[i], '.3f')  + "\n"
         
    # B Bank
    for i in range(1, 61):
        if (i<10):
            str_output += "Leaf  " + str(i) + "B = " + format(B_pos[i], '.3f') + "\n"
        else:
            str_output += "Leaf " + str(i) + "B = " + format(B_pos[i], '.3f')+ "\n"

    str_output += "Note = 0\nShape = 0\nMagnification = 1.00\n"

print(str_output)




print("Increment: " + str(increment))

print(filename)

f = open(filename, "w")
f.write(str_output) 
f.close()