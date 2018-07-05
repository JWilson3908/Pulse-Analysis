# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:36:55 2018

@author: Dima
    0 - date nan;     1 - time nan;    2 - Universal time;    3 - X(V);    4 - Y(V)
    5 - Drive Frequency; 6- Q;    7 - Inferred frequency;    8 - X - feedthrough;    9 - Y - feedthrough
    10 - Drive voltage;    11 - k_eff;    12 - R_dale;    13 - T_mct;     14 - c_mct
    15  - T_pa;    16 - SP;    17 - BP;    18 - WT
"""

#%% Initializing Section

import matplotlib.pyplot as plt
import numpy as np
from johnsFunctions import chooseFile
#import sys

#%% chooses a path and removes pulses

path = chooseFile()


# previously the path was "d:\\therm_transport\\thermal_cond\\FF2mKheat.dat"
# updated to open a dialog box via a function I wrote - John

num_exp=20 # number of point around pulse to remove

data=np.genfromtxt(path,
                   unpack=True,
                   skip_header=1,
                   usecols = (2, 6, 13, 7)
                   )

""" 0 - time; 1 - frequency; 2 - Q; 3 - Tmct"""

data[0] = data[0]-data[0][0] 

a = np.where( abs( data[2] ) > 2000 )[0]
    # np.where function creates a vector of points greater than 2000 in Q
b = set()
    # this creates a set of numbers to use for b

for i in a:
    
    for j in range( -num_exp, num_exp ):

        b.add( i + j )
        
        # for a point in i, j points to either side are recorded in b
        # Using the add function instead of append skips double numbers
        
b = list( b )
    # leaving b as a set messes up later code
    
d = np.in1d( range( 0, len(data[0]) ), b, invert = True)
    # creates a list of points that dont have pulsing near them
    # no need for the assume unique state anymore
    
#%% Creates new lists of data with pulsing now removed
Q=data[2][d]
time=data[0][d]
F=data[4][d]
T=data[3][d]

#%% Plotting

fig1 = plt.figure(1, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Q')
ax1.set_xlabel('time')
ax1.set_title('Q vs time')
line, = ax1.plot(time, Q, color='blue', lw=2)
plt.show()

fig2 = plt.figure(2, clear = True)
ax2 = fig2.add_subplot(111)
ax2.set_ylabel('Q')
ax2.set_xlabel('temperature')
ax2.set_title('Q vs temperature')
line, = ax2.plot(T, Q, color='blue', lw=2)
plt.show()