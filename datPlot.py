# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:38:20 2018

@author: John Wilson

This script asks for a set of data (expecting the FF .dat files from
the Parpia Lab) and plots your Q versus time
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tkn

def chooseFile():

    tkn.Tk().withdraw() # Close the root window
    in_path = tkn.filedialog.askopenfilename() # This opens a dialog box
    return(in_path) # This gives us the output of the chooseFile function
    
def datPlot():
    # Note that using chooseFile() removes the need for an r before
    # the string in genfromtxt
    filename = chooseFile()
    data = np.genfromtxt(filename,
                         skip_header=1,
                         skip_footer=1,
                         unpack=1,
                         usecols=(2,6)
                         )
    plt.plot(data[0],data[1])
    
                    # Enable to two functions below to set the y-axis
                    # 150 is upper bound for SF data.
                    
    axes = plt.gca()
    axes.set_ylim([0,150])
    
    plt.show()

datPlot()