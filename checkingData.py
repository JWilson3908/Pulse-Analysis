# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:20:11 2018

@author: John Wilson
"""
# -*- coding: utf-8 -*-

import numpy as np
from tkinter import filedialog #This prevents an error that occurs occasionally
import tkinter as tkn
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
                            usecols=(2,6,13)
                            )

    c=1
    
    while c < days :
        filename = chooseFile()
        newdata = np.genfromtxt(filename,
                                skip_header=1,
                                unpack=1,
                                usecols=(2,6,13)
                                )
        bar=[[newdata[0][len(newdata-1)]],[-275000],[-275000]]
    
        fullData=np.concatenate((fullData,bar,newdata),axis=1)
        c=c+1
        
    else :
        print(min(fullData[2]))
        print(max(fullData[2]))
        return(fullData)
    
def datPlot(days,col):
    # Col is used to choose the data column the within the FF.dat file
    
    data=combineData(days)
    xs = data[0]
    ys = data[col]
    
    plt.plot(
            xs,ys,
            marker = None
            )
    
    if col == 1 :
        plt.title('Q vs Time')
        
        axes = plt.gca()
        axes.set_ylim([-20,200])
        
    elif col == 2:
        
#        if np.all( ys <= 5 ):
            axes = plt.gca()
            axes.set_ylim([0,1])
            plt.title('Temperature vs Time')
            
#        else :
 #           axes = plt.gca()
  #          axes.set_ylim([-20,30])
   #         plt.title('Temperature vs Time with High Temperature')
            
        
    plt.show()
    
datPlot(1,1)
