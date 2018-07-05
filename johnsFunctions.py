# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:21:12 2018

@author: John Wilson

First attempt at creating a module to add funcitons to Dima's code
"""
#%% importing necessary modules
from tkinter import filedialog as fd 
# Importing like this prevents an error that occurs occasionally
import tkinter as tkn
import numpy as np

#%% Defining functions to be called by other codes

def chooseFile():

    tkn.Tk().withdraw() # Close the root window
    in_path = fd.askopenfilename()
    return(in_path)
    
def chooseDir() :
    tkn.Tk().withdraw() # Close the root window
    in_dir = fd.askdirectory()
    return(in_dir)
    
def combineData(days):
    
    filename = chooseFile()
    fullData = np.genfromtxt(filename,
                            skip_header=1,
                            unpack=1,
                            usecols=(2, 6, 13, 7)
                            )

    c=1
    
    while c < days :
        filename = chooseFile()
        newdata = np.genfromtxt(filename,
                                skip_header=1,
                                unpack=1,
                                usecols=(2, 6, 13, 7)
                                )
    
        fullData=np.concatenate((fullData,newdata),axis=1)
        c=c+1
        
    else :
        return(fullData)
        
def tempC(Ppsi) :
    
    Pbar = Ppsi * 0.0689476
    
    a0 = 0.92938375
    a1 = 0.13867188
    a2 = -0.0069302185
    a3 = 0.00025685169
    a4 = -0.0000057248644
    a5 = 0.000000053010918
    
    tc = a0 + a1 * (Pbar) + a2 * (Pbar**2) + a3 * (Pbar**3) + a4 * (Pbar**4) + a5 * (Pbar**5)
    
    return(tc)
        
def innum():
    # This block of code is not done yet
    
    try: d #This prevents data from being needlessly reloaded
    except NameError: d = combineData(1)

    nums = [np.float64( range( 1, len( d[0] ) + 1 ) )]
    data = np.concatenate( ( d, nums), 0 )
    
    print(data[4][134])

#%% Graveyard of code
        
#def datPlot(days):
#    
#    data=combineData(days)
#    
#    p1 = plt.plot( data[0], data[1] )
#    
#    plt.show( p1 )
#    
#    return(data)
#    
#data = datPlot(1)
#
#dble=np.float64([[0],[0]])
#
#dx = np.diff( data[0] )
#
#dy = np.diff( data[1] )
#
#dy=np.concatenate( (dy,[1]) )
#dx=np.concatenate( (dx,[1]) )
#
#arr=np.float64([data[0],(np.divide(dy,dx))])
#
#dp1 = plt.plot( data[0], ( dy / dx ) )
#
#plt.show( dp1 )
#
# for i in np.arange( 15 , len( data[0] ) - 15 ):
# The above line is the remenents of a different approach

combineData(4)



