# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:24:18 2018

@author: John Wilson
"""

from pathlib import Path
import numpy as np

data_folder = Path("C:/Users/John Wilson/Documents/Cornell REU/Data/All Data/")
files = ["20180315\\FF0p7mK.dat","20180316\\FF0p8mK.dat","20180317\\FF0p9mK.dat"]

file_to_open= data_folder / files[1]

f = open(file_to_open)

data = np.genfromtxt(f,
                     skip_header=1,
                     unpack=1,
                     usecols=(2, 6, 13, 7)
                     )
for i in range(2):
    
    file_to_open= data_folder / files[i+1]

    f = open(file_to_open)


    newdata = np.genfromtxt(f,
                            skip_header=1,
                            unpack=1,
                            usecols=(2, 6, 13, 7)
                            )
    data = np.concatenate( (data, newdata), axis=1)

del(newdata)
    