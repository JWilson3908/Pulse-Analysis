# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:38:56 2018

@author: John Wilson
"""

import tkinter as tkn

def chooseFile():

    tkn.Tk().withdraw() # Close the root window
    in_path = tkn.filedialog.askopenfilename() # This opens a dialog box
    return(in_path) # This gives us the output of the chooseFile function