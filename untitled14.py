# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:56:46 2018

@author: John Wilson
"""
import numpy as np
import matplotlib.pyplot as plt
import johnsFunctions as jw
import sys
    
try: d1 #This prevents data from being needlessly reloaded
except NameError: d1 = jw.combineData(1)

try: d2 #This prevents data from being needlessly reloaded
except NameError: d2 = jw.combineData(1)

# filterPulses(self,d1,d2,num_exp1,num_exp2) :

num_exp1 = 10
num_exp2 = 100

d1[0] = d1[0]-d1[0][0]
        
d2[0] = d2[0]-d2[0][0] 

a = np.where( abs( d2[1] ) > 1500 )[0]
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

d = np.isin( range( 0, len(d2[0]) ), b, invert = True)
# creates a list of points that dont have pulsing near them
# no need for the assume unique state anymore

time=d2[0][d]
Q=d2[1][d]
F=d2[3][d]
T=d2[2][d]
treatedData2 = np.float64([time,Q,F,T])
treatedData1 = np.zeros((4,len(treatedData2[0])))

for i in range(4):

    treatedData1[i] = d1[i][d]


n1s = [np.float64( range( 1, len( treatedData1[0] ) + 1 ) )]
n2s = [np.float64( range( 1, len( treatedData2[0] ) + 1 ) )]

treatedData1 = np.concatenate( ( treatedData1, n1s), 0 )
treatedData2 = np.concatenate( ( treatedData2, n2s), 0 )   

#fig1 = plt.figure(1, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Q')
#ax1.set_xlabel('time')
#ax1.set_title('Q vs time')
#line, = ax1.plot(treatedData2[0], treatedData2[1], color='blue', lw=2)
#plt.show()         


     
#    def temp_fit(self,nump):
'''linear regression fit of temperature data, removing nan first'''
deg = 1

na=np.where(np.isnan(d1[2]))
ddd=np.isin(range(0,len(d1[2])),na,invert = True)

fit = np.polyfit(d1[0][ddd],d1[2][ddd],deg)
fit_fn = np.poly1d(fit) 
print(fit_fn)
#w=np.ones(len(d1[2]))
#w[int(len(w)/2):]=2

#,w=w[ddd]


temp2=fit_fn(d1[0][d & ddd])
dt=jw.tempC(425)-np.mean(temp2[-30:-1]) #correction to tc
fit[-1]+=dt
temp2=fit_fn(d1[0][d & ddd])
fit_rev=np.polyfit(temp2,d1[0][d & ddd],deg)
timeRev=np.poly1d(fit_rev)


fig1 = plt.figure(7, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('T')
ax1.set_xlabel('time')
ax1.set_title('T and time')

ax1.plot(d1[0][ddd], d1[2][d & ddd], color='green',lw=1)

ax1.plot(d1[0][d & ddd], fit_fn(d1[0][d & ddd]), color='blue',lw=1)

plt.grid()
plt.show()

fig1 = plt.figure(8, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('time')
ax1.set_xlabel('T')
ax1.set_title('T and time reverse')
ax1.plot(d1[2][d], d1[0][d], color='green',lw=1)
ax1.plot(temp2, timeRev(temp2), color='blue',lw=1)
plt.grid()
plt.show()
    
fit1=tuple(fit)
fit_rev1=tuple(fit_rev)

