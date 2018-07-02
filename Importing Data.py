# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:38:20 2018

@author: John Wilson
"""
# This script exists to provide a template for directly using strings to open
# files containing data

import numpy as np
import matplotlib.pyplot as plt

""" filename = open ( r ) """
from pathlib import Path

filename = Path(r'C:\Users\John Wilson\Documents\Cornell REU\Data\S.F. Practice Data\201802_09+10\FF Combined Data.dat')
data = np.genfromtxt(filename,
                     skip_header=1,
                     skip_footer=1,
                     unpack=1,
                     usecols=(2,7)
                     )
plt.plot(data[0],data[1])

plt.show()