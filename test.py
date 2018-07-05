# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:42:33 2018

@author: John Wilson
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
#import sys
import time as e_t
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter('ignore', np.RankWarning)

import johnsFunctions as jw

def importData(path):    
    
    data = np.genfromtxt(path[0],
                         skip_header=1,
                         unpack=1,
                         usecols=(2, 6, 13, 7)
                         )
    
    for i in range( len(path) - 1):
        newdata = np.genfromtxt(path[i],
                         skip_header=1,
                         unpack=1,
                         usecols=(2, 6, 13, 7)
                         )
        data = np.concatenate((data,newdata),axis=1)
        
    return(data)

def filterPulse(dataset,num_exp1,num_exp2) :
    
    a = np.where( abs( dataset[1] ) > 2000 )[0]
    # np.where function creates a vector of points greater than 2000 in Q
    b = set()
    # this creates a set of numbers to use for b

    for i in a:
    
        for j in range( -num_exp1, num_exp2 ):
            
            b.add( i + j )
        
            # for a point in i, j points to either side are recorded in b
            # Using the add function instead of append skips double numbers
        
    b = list( b )
        # leaving b as a set messes up later code
    
    d = np.in1d( range( 0, len(dataset[0]) ), b, invert = True)
        # creates a list of points that dont have pulsing near them
        # no need for the assume unique state anymore
    
    Q=dataset[1][d]
    time=dataset[0][d]
    F=dataset[3][d]
    T=dataset[2][d]
    treatedData = np.float64([time,Q,F,T])
    return(treatedData,d)

P = 0.6205284 # Enter your pressure in bar

tc = jw.tempC(P) # Finds Tc for that pressure

impDir="C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        
# Fork 1
path1=[impDir+"20180315\\CF0p7mK.dat",impDir+"20180316\\CF0p8mK.dat",impDir+"20180317\\CF0p9mK.dat"]
            
# Fork 2
path2=[impDir+"20180315\\FF0p7mK.dat",impDir+"20180316\\CF0p8mK.dat",impDir+"20180317\\FF0p9mK.dat"]

rawdata1 = importData(path1)
rawdata2 = importData(path2)

rawdata1[0] = rawdata1[0]-rawdata1[0][0] 
rawdata2[0] = rawdata2[0]-rawdata2[0][0] 

[data2,d] = filterPulse(rawdata2,10,100)

data1=np.zeros((4,len(data2[0])))

for i in range(4):
    
    data1[i] = rawdata1[i][d]

# Code graveyard
#
#fig1 = plt.figure(1, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Q')
#ax1.set_xlabel('time')
#ax1.set_title('Q vs time')
#line, = ax1.plot(data2[0], data2[1], color='blue', lw=2)
#plt.show()
#
#
#fig2 = plt.figure(2, clear = True)
#ax2 = fig2.add_subplot(111)
#line, = ax2.plot(rawdata2[0], rawdata2[1], color='red', lw=2)
#plt.show()
#
#
#fig2 = plt.figure(2, clear = True)
#ax2 = fig2.add_subplot(111)
#ax2.set_ylabel('Q')
#ax2.set_xlabel('temperature')
#ax2.set_title(' Q vs temperature ')
#line, = ax2.plot(data2[3], data2[1], color='blue', lw=2)
#plt.show()
