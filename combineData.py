# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:50:28 2018

@author: John Wilson
"""
import numpy as np
import tkinter as tkn
import tkinter.filedialog # Stack overflow identified an error that sometimes
                          # causes filedialog not to operate correctly
import matplotlib.pyplot as plt


def chooseFile():

    tkn.Tk().withdraw() # Close the root window
    in_path = tkn.filedialog.askopenfilename()
    return(in_path)


def combineData(days):
    
    filename = chooseFile()
    fullData = np.genfromtxt(filename,
                            skip_header=1,
                            unpack=1,
                            usecols=(2,6)
                            )

    c=1
    
    while c < days :
        filename = chooseFile()
        newdata = np.genfromtxt(filename,
                                skip_header=1,
                                unpack=1,
                                usecols=(2,6)
                                )
    
        fullData=np.concatenate((fullData,newdata),axis=1)
        c=c+1
        
    else :
        return(fullData)
    
def datPlot(days):
    
    data=combineData(days)
    
    p1 = plt.plot( data[0], data[1] )
    
    plt.show( p1 )
    
    return(data)
    
data = datPlot(1)

dble=np.float64([[0],[0]])

dx = np.diff( data[0] )

dy = np.diff( data[1] )

dy=np.concatenate( (dy,[1]) )
dx=np.concatenate( (dx,[1]) )

arr=np.float64([data[0],(np.divide(dy,dx))])

dp1 = plt.plot( data[0], ( dy / dx ) )

plt.show( dp1 )

# for i in np.arange( 15 , len( data[0] ) - 15 ):
# The above line is the remenents of a different approach
    



