# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:44:46 2018

@author: John Wilson
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:01:42 2018

@author: John Wilson

"""
import numpy as np
import matplotlib.pyplot as plt

def tempC(Ppsi) :
    ''' This function takes our direct reading of pressure (in psi) and gives us the expected Tc'''
    Pbar = Ppsi * 0.0689476
    # Greywall Paper Coeffecients
    a0 = 0.92938375
    a1 = 0.13867188
    a2 = -0.0069302185
    a3 = 0.00025685169
    a4 = -0.0000057248644
    a5 = 0.000000053010918
    
    tc = a0 + a1 * (Pbar) + a2 * (Pbar**2) + a3 * (Pbar**3) + a4 * (Pbar**4) + a5 * (Pbar**5)
    
    return(tc)

def finalPlot(Pulse):
    '''This function plots final data of pulses'''
    fig1 = plt.figure(3000, clear = True)
    ax1 = fig1.add_subplot(111)
    ax1.set_ylabel('Temperature (mK)')
    ax1.set_xlabel('time')
    ax1.set_title('Pulsing Temperature vs time')
    ax1.scatter(Pulse.pulserawData2[0], Pulse.pulserawData2[5], color='blue',s=0.5)
    axes = plt.gca()
    axes.set_ylim([2.14,max(Pulse.pulserawData2[5])])
    plt.show()
    
class pulse:
    def __init__(self, Ppsi, *nums):
        # Input arguments are defined here
        self.Ppsi = Ppsi
        self.Pbar = self.Ppsi * 0.0689475729
        self.Tc = tempC(self.Ppsi)
        self.ramp = nums[0]
        # Ramp is either 1 or -1 for warming or cooling.
        # This is used to reverse the order of data in warming so warming
        # and cooling ramps can be treated the same
        self.doneCutting = nums[1]
        # This is used to alternate between plots being displayed. 
        # Prior to data treatment, during the cutting stage this is 1 and everything
        # will be plotted against point #'s
        self.basestart = nums[2]
        self.basestop = nums[3]
        self.pulsestart = nums[4]
        self.pulsestop = nums[5]
        # Initial arguments for cutting
        self.ne1 = nums[6]
        self.ne2 = nums[7]
        # Number of points to be removed before/after pulses
        self.date = nums[8]
        # Date used to choose which data file to import into the path
        self.impDir = "C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        # This import directory should contain all the folders that contain the date(s)
        # you want to look at
        if self.date == 602 :
            # Fork 1
            self.basepath1 = [self.impDir+"20180602\\CF2p3mK.dat",self.impDir+"20180603\\CF2p2mK.dat"]
            # Fork 2
            self.basepath2 = [self.impDir+"20180602\\FF2p3mK.dat",self.impDir+"20180603\\FF2p2mK.dat"]

            # Fork 1 
            self.pulsepath1 = [self.impDir+"20180604\\CF2p2mK.dat",self.impDir+"20180605\\CF2p1mK.dat",self.impDir+"20180606\\CF2p1mK.dat",self.impDir+"20180607\\CF1p8mK.dat",self.impDir+"20180608\\CF1p9mK.dat",self.impDir+"20180609\\CF2p0mK.dat",self.impDir+"20180610\\CF2p0mK.dat"]
            # Fork 2
            self.pulsepath2 = [self.impDir+"20180604\\FF2p2mK.dat",self.impDir+"20180605\\FF2p1mK.dat",self.impDir+"20180606\\FF2p1mK.dat",self.impDir+"20180607\\FF1p8mK.dat",self.impDir+"20180608\\FF1p9mK.dat",self.impDir+"20180609\\FF2p0mK.dat",self.impDir+"20180610\\FF2p0mK.dat"]
            
        self.baserawData1,self.baserawData2 = self.importData(self.basestart,self.basestop,self.basepath1),self.importData(self.basestart,self.basestop,self.basepath2) # import fork 1, fork 2 
        self.pulserawData1,self.pulserawData2 = self.importData(self.pulsestart,self.pulsestop,self.pulsepath1),self.importData(self.pulsestart,self.pulsestop,self.pulsepath2) # import fork 1, fork 2 
        
        if self.doneCutting == 0 :    
            
            n1s = [np.float64( range( 1, len( self.baserawData1[0] ) + 1 ) )]
            self.baserawData1 = np.concatenate( (self.baserawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.baserawData2[0] ) + 1 ) )]
            self.baserawData2 = np.concatenate( (self.baserawData2, n2s), 0 )
            self.cutFunction('base')
            
            n1s = [np.float64( range( 1, len( self.pulserawData1[0] ) + 1 ) )]
            self.pulserawData1 = np.concatenate( (self.pulserawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.pulserawData2[0] ) + 1 ) )]
            self.pulserawData2 = np.concatenate( (self.pulserawData2, n2s), 0 )
            self.cutFunction('pulse')
            
        elif self.doneCutting == 1:            
            if self.ramp == 1:
                for i in range( 1, len( self.baserawData1 ) - 1  ) :
                    self.rawData1[i] = self.baserawData1[i][::-1]                
                for i in range( 1, len( self.baserawData2 )  ) :
                    self.rawData2[i] = self.baserawData2[i][::-1]
            n1s = [np.float64( range( 1, len( self.baserawData1[0] ) + 1 ) )]
            self.baserawData1 = np.concatenate( (self.baserawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.baserawData2[0] ) + 1 ) )]
            self.baserawData2 = np.concatenate( (self.baserawData2, n2s), 0 )  
            n1s = [np.float64( range( 1, len( self.pulserawData1[0] ) + 1 ) )]
            self.pulserawData1 = np.concatenate( (self.pulserawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.pulserawData2[0] ) + 1 ) )]
            self.pulserawData2 = np.concatenate( (self.pulserawData2, n2s), 0 )
            self.d, self.dtemp = self.filterPulses()            
            # dtemp gives a logic string for removing NaN from temperature data
            # dtemp also removes points with pulsing near by                                    
        
    def importData(self,start,stop,path):        
        ''' Function for importing the data in a given path'''
        fulldata = np.genfromtxt(path[0],
                                 skip_header=1,
                                 unpack=1,
                                 usecols=(2, 6, 13, 7)
                                 )
        for i in range( len(path) - 1):
            newdata = np.genfromtxt(path[i + 1],
                                    skip_header=1,
                                    unpack=1,
                                    usecols=(2, 6, 13, 7)
                                    )
            fulldata = np.concatenate( (fulldata, newdata), axis=1)        
        data = fulldata[0:,start:stop] 
        # self.start and stop defined by the initialization of the class.
        # These are meant to be used in junction with the self.doneCutting variable
        # to make the cutFunction display plots that will help cutting go faster
        t0=data[0][0]        
        data[0]=data[0]-t0    
        return(data)
    
    def cutFunction(self,BaseOrPulse):        
        '''This function plots Temperature, Fork 1 Q, and Fork 2 Q vs Point #
         to help with cutting. start and stop points should be changed with 
         this function being used to plot inbetween changes '''
         
        if BaseOrPulse == 'base' :
            fig1 = plt.figure(1, clear = True)
            ax1 = fig1.add_subplot(131)
            ax1.set_ylabel('Tmc (mK)')
            ax1.set_title('Tmc and Q(F1 & F2) vs Point # for cutting for '+ str(self.date) + ' ' + 'base')
            line, = ax1.plot(self.baserawData1[4] + self.basestart, self.baserawData1[2], color='blue', lw=2)
            ax1 = fig1.add_subplot(132)
            ax1.set_ylabel('Fork 1 Q')
            ax1.scatter( self.baserawData1[4]+self.basestart, self.baserawData1[1], color='blue', s=0.5)
            axes = plt.gca()
            axes.set_ylim([0,max(self.baserawData1[1])])
            ax1 = fig1.add_subplot(133)
            ax1.set_ylabel('Fork 2 Q')
            ax1.set_xlabel('Point #')
            ax1.scatter( self.baserawData2[4]+self.basestart, self.baserawData2[1], color='blue', s=0.5)
            axes = plt.gca()
            axes.set_ylim([0,max(self.baserawData2[1])])
            plt.show()
        
        elif BaseOrPulse == 'pulse' :
            fig1 = plt.figure(2, clear = True)
            ax1 = fig1.add_subplot(131)
            ax1.set_ylabel('Tmc (mK)')
            ax1.set_title('Tmc and Q(F1 & F2) vs Point # for cutting for '+ str(self.date) + ' ' + 'pulse')
            line, = ax1.plot(self.pulserawData1[4] + self.pulsestart, self.pulserawData1[2], color='blue', lw=2)
            ax1 = fig1.add_subplot(132)
            ax1.set_ylabel('Fork 1 Q')
            ax1.scatter( self.pulserawData1[4]+self.pulsestart, self.pulserawData1[1], color='blue', s=0.5)
            axes = plt.gca()
            axes.set_ylim([0,max(self.pulserawData1[1])])
            ax1 = fig1.add_subplot(133)
            ax1.set_ylabel('Fork 2 Q')
            ax1.set_xlabel('Point #')
            ax1.scatter( self.pulserawData2[4]+self.pulsestart, self.pulserawData2[1], color='blue', s=0.5)
            axes = plt.gca()
            axes.set_ylim( [ 0, max(self.pulserawData2[1]) ])
            plt.show()
        
    def filterPulses(self) :
        ''''This function filters pulses when pulsing occurs in fork 2. It hasn't been vigurously tested yet as of 07/13/2018'''
        a = np.where( self.pulserawData2[1] < 0 )[0]
        self.pulseIndex = []
        for i in range(1,len(a)):
            if a[i] - a[i-1] > 1 :
                self.pulseIndex.append(a[i-1])
        
        # np.where function creates a vector of points greater than 1500 in Q
        b = set()
        # this creates a set of numbers to use for b, sets do not allow repeat entries
        for i in a:
            for j in range( -self.ne1, self.ne2 ):
                b.add( i + j )
                # for a point in i, j points to either side are recorded in b
                # Using the add function instead of append skips double numbers
        b = list( b )
        # converts b to a list so it works with np.isin function
        d = np.isin( range( 0, len(self.pulserawData2[0]) ), b, invert = True)
        na = np.where(np.isnan(self.baserawData1[2]))
        # na is a list of NaN points in our Tmc.
        # This is especially useful under 1 mK data when labview is likely to generate NaN Temperatures
        dtemp = np.isin(range(0,len(self.baserawData1[2])),na,invert = True)    
        # creates a list of points that dont have pulsing near them, and removes NaN
        # no need for the assume unique state anymore because of our use of b being a set
        fig1 = plt.figure(1, clear = True)
        # Creating a plot of both fork's Qs to check that pulsing has been removed
        ax1 = fig1.add_subplot(211)
        ax1.set_ylabel('Fork 1 Q')
        ax1.set_title('Q(F1) and Q(F2) vs Time for Pulse Removal for '+ str(self.date) + ' Pulsing')
        ax1.scatter( self.pulserawData1[0][d], self.pulserawData1[1][d], color='blue', s=0.5)
        ax1 = fig1.add_subplot(212)
        ax1.set_ylabel('Fork 2 Q')
        ax1.set_xlabel('Time (s)')
        ax1.scatter( self.pulserawData2[0][d], self.pulserawData2[1][d], color='blue', s=0.5)
        plt.show()
        return d, dtemp
    
    def temp_fit(self,deg):
        '''This function fits melting curve temperature and adds an offset so it
         can be used as the local temperature for Fork 1'''
        w = np.ones(len(self.baserawData1[2]))
#        w[int(len(w)/2):]=2
        # Adjust w to adjust the weighting of the fit, if left commented there will be no weighting
        fit = np.polyfit(self.baserawData1[0][self.dtemp],self.baserawData1[2][self.dtemp],deg,w=w[self.dtemp])
        # Generates a fit from Tmc as a function of time
        fit_fn = np.poly1d( fit )
        fitTemp = fit_fn(self.baserawData1[0][self.dtemp])
        # Generates a set of temperatures from a time input
        dt = self.Tc - np.mean(fitTemp[1:30])        
        self.dt = dt
        # Takes the difference between the theoretical Tc and the point we cut to (Which should be the real Tc)
        fit[-1] += dt # The last element of the fit (constant offset) is shifted by dt
        fitTemp = fit_fn(self.baserawData1[0][self.dtemp])
        # This generates a new set of temperatures, this time shifted to the real Tc
        fig1 = plt.figure(2, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T (mK)')
        ax1.set_xlabel('time (s)')
        ax1.set_title('Tmc and T F1 (corrected fit) vs time for '+ str(self.date))
        ax1.plot(self.baserawData1[0][self.dtemp], self.baserawData1[2][self.dtemp], color='green',lw=1)
        ax1.plot(self.baserawData1[0][self.dtemp], fit_fn(self.baserawData1[0][self.dtemp]), color='blue',lw=1)
        plt.grid()
        plt.show()
        self.baserawData1 = np.vstack(( self.baserawData1, fit_fn(self.baserawData1[0][self.dtemp]) ))
        # np.vstack is used to save the new (local) temperature for Fork 1 into the self.baserawData1 array
        # Row 5 (index 4) is now the corrected temperature for baserawData1 (Fork 1)
        # Row 3 (index 2) is still the Tmc temperature (in both baserawData1 and baserawData2)        
        fit1 = tuple(fit)
        # fit saved as tuple to avoid accidental changes to the fit
        self.T_fit = fit1 # saved as a self. variable to make it easier to see the fit later
        
    def QtotimeF1(self,npol1):
        ''' Fit Fork 1 with a polynomial + step function for Q(t) '''
        tempData1 = self.baserawData1[1] 
        # Creates a temporary version of Q values within fork 1
        #  that we can adjust without fear of losing our original data
        self.timeAB = 0 # Initializes our variables for the step function because
        self.step = 0   # Python doesn't like to fit non-linear models (i.e. Stepfunctions)
        if self.Pbar >= 21.22 :
            # If Pbar is below the pcp then there will be no AB transition to identify in fork 1
            # Date 705 is also excluded because CF has no identifiable AB in that data range
            for i in range( 40, len(tempData1) - 40 ): # Identify the AB transition
                # Range doesn't actually matter, it won't lead to the function
                # calling a point for averaging thats outside the index
                avg_1 = sum(tempData1[(i + 1):(i + 21)]) / np.float64(len(tempData1[(i + 1):(i + 21)]))
                avg_2 = sum(tempData1[(i - 20):i]) / np.float64(len(tempData1[(i - 20):i]))
                # Averages are computed to avoid noise from being flagged as the AB transition
                if abs(avg_1 - avg_2) > abs(self.step) :
                    self.step = avg_1 - avg_2
                    # Step is set to the difference between these two
                    self.pointAB = self.baserawData1[4][i] # Point and time are saved as self. variables to help
                    self.timeAB = self.baserawData1[0][i]  # us identify AB later, outside of Python if needed
                    
        tempData1 = tempData1 - self.step * np.heaviside( self.baserawData1[0] - self.timeAB, 0 )    
        # subtract off the heavisidetheta function and fit the remaining data with a polynomial
        # This (hopefully) keeps Python from trying to fit a discontinuous function
        w = np.ones(len(tempData1))
#        w[0:60] = 5
        # Turn on weighting here to givea higher weight to early points near Tc
        # Adjust w to adjust the weighting of the fit
        qfit = np.polyfit(self.baserawData1[0][self.dtemp],tempData1[self.dtemp],npol1,w = w[self.dtemp])
        # This generates fit of the temporary data to the original time, then the
        # step function can be added into that fit later so it works on our baserawData1
        qfit = tuple(qfit)
        # Saving the fit as a tuple to avoid it accidentally being overwritten
        qfit_fn = np.poly1d(qfit) # Q
        # Generates a fit for the temporary data 
        Q = qfit_fn(self.baserawData1[0][self.dtemp])
        # This generates data based off our fit from the temporary data
        Qplot = Q + self.step * np.heaviside( self.baserawData1[0] - self.timeAB, 0 )
        # Adding in the step function to be able to compare it to the true data
        # Plotting to check if Q fit to time is good        
        fig1 = plt.figure(3, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 1\'s Q and Q fitted vs point # ( Both from Fork 1 ) for ' + str(self.date) + ' Base ' )
        ax1.scatter(self.baserawData1[4][self.dtemp], self.baserawData1[1][self.dtemp], color='blue',s=0.5)
        ax1.plot(self.baserawData1[4][self.dtemp], Qplot, color='red',lw=1)
        plt.grid()
        plt.show()
        print(len(self.baserawData1))
        self.baserawData1 = np.vstack((self.baserawData1,Qplot))
        print(len(self.baserawData1))
        self.fit_QtoF1 = qfit
        # Saves as a self. variable to make it easier to see and use the fit later
        
    def reCallibrateF2(self,shift):
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        Q = qfit_fn(self.baserawData1[0][self.dtemp])
        Qplot = Q + self.step * np.heaviside( self.baserawData1[0] - self.timeAB, 0 )
        # Generates data to be plotted from the fit of our original Q in fork 1.
        self.p0 = shift + 1
        # plus one here to make sure we don't mess up any indexing later on
        # p0 is saved as an attribute to the class because its later used
        # to save time in indexing within a binary search
        q0 =  Qplot[0] - self.baserawData2[1][shift]            
        # q0 created to show the shift entered into the command
        self.baserawData2[4] = self.baserawData2[4] - self.p0
        # We can edit rawData2[4] because it is a list of numbers we generated anyway
        self.baserawData2 = np.vstack((self.baserawData2, self.baserawData2[1] + q0 ))
        # Add new rows for corrected q value to be used later in the k-NN search
        # Plots the shifted Fork 2 data to compare with the Fork 1 fit and match Tc points
        fig1 = plt.figure(4, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 2 Q vs point # and Fork 1 fit for '+ str(self.date))
        ax1.scatter(self.baserawData2[4], self.baserawData2[5], color='blue',s=0.5)
        line, = ax1.plot(self.baserawData1[4], Qplot, color='red', lw=1)
        ax1.annotate('Tc points should match here', xy=(0, Qplot[0]), xytext=(self.baserawData2[4][-500], self.baserawData2[5][500]))
        # xytext point needs to be adjusted sometimes because the index location might be too big, or not big enough if it covers data
        plt.show()    
    
    def k_NNsearch(self,tol=0.05):
        '''Enter a tolerance and the search will carry out a closest nearest neighbor search,
         implimented via a binary search to convert Q in fork 2 to Temperature'''
        # Print functions are used heavily in this function because the
        # program is prone to getting stuck inside this function.
        print('Starting k-NN Search')
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        Q = qfit_fn(self.baserawData1[0][self.dtemp])
        u = 0
        # If an AB point occurs in CF we make sure to use the step function fit
        if self.Pbar >= 21.22:
            u = self.step * np.heaviside( self.baserawData1[0] - self.timeAB, 9000 )
            # The sceond argument of np.haeviside changed to let us skip over the point
            # where the AB transition occurs, otherwise the function would get stuck there
            print('setting step function')
        Qplot = Q + u
        newTemp = np.full(len(self.baserawData2[5]), -0.5)
        # Since temperature is never negative, intializing an array this way
        # will let us skip these points later on
        for i in range(int(self.p0), len(self.baserawData2[5])):
            first = 0
            last = len( self.baserawData2[5] ) - 1
            a = True
            if Qplot[i] > 9000 or self.baserawData2[5][i] < Qplot[0]:
                # This skips the intial points that have no Q value to map to,
                # as well as the one point on the CF AB transition
                newTemp[i] = newTemp[i-1]
                a = False
                
            if self.baserawData2[5][i] > Qplot[-1]:
                a = False
                #This skips points after the last Qplot point
            while a == True:
                m = (first + last) / 2
                m = int(m)
                # This sets the midpoint for the search
                # Inclue this in the if in order to use a tolerance for the k-NN search
                # abs( Qplot[m] - self.baserawData2[5][i] ) < tol
                if abs(first - last) <= 3 :
                    newTemp[i] = self.baserawData1[5][m]
                    a = False
                    del m
                    print(i, 'completed')
                elif Qplot[m] > self.baserawData2[5][i] :
                    last = m + 1
                elif Qplot[m] < self.baserawData2[5][i]:
                    first = m - 1
        
        self.baserawData2 = np.vstack(( self.baserawData2, newTemp ))

    def tempAdjustPulses(self):
        '''Enter a tolerance and the search will carry out a closest nearest neighbor search,
         implimented via a binary search to convert Q in fork 2 to Temperature'''
        # Print functions are used heavily in this function because the
        # program is prone to getting stuck inside this function.
        print('Starting k-NN Search on Pulses')
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        Q = qfit_fn(self.baserawData1[0][self.dtemp])
        u = 0
        # If an AB point occurs in CF we make sure to use the step function fit
        if self.Pbar >= 21.22:
            u = self.step * np.heaviside( self.baserawData1[0] - self.timeAB, 9000 )
            # The sceond argument of np.haeviside changed to let us skip over the point
            # where the AB transition occurs, otherwise the function would get stuck there
            print('setting step function')
        Qplot = Q + u
        newTemp = np.full(len(self.pulserawData2[1]), -0.5)
        # Since temperature is never negative, intializing an array this way
        # will let us skip these points later on
        for i in range(0, len(self.pulserawData2[1])):
            first = 0
            last = len( Qplot )
            a = True
            if self.pulserawData2[1][i] < min(Qplot) or self.pulserawData2[1][i] > max(Qplot) :
                # This skips the intial points that have no Q value to map to,
                # as well as the one point on the CF AB transition
                newTemp[i] = newTemp[i-1]
                a = False
                
#            if self.pulserawData2[1][i] > Qplot[-1]:
#                a = False
#                #This skips points after the last Qplot point
            while a == True:
                m = (first + last) / 2
                m = int(m)
                # This sets the midpoint for the search
                # Inclue this in the if in order to use a tolerance for the k-NN search
                # abs( Qplot[m] - self.baserawData2[5][i] ) < tol
                if abs(first - last) <= 3 :
                    newTemp[i] = self.baserawData1[5][m]
                    a = False
                    del m
                    print(i, 'completed')
                elif Qplot[m] > self.pulserawData2[1][i] :
                    last = m + 1
                elif Qplot[m] < self.pulserawData2[1][i]:
                    first = m - 1
        self.pulserawData2 = np.vstack(( self.pulserawData2, newTemp )) 
        
    def findABs(self,tol) :
        self.TimeABs=np.zeros(len(self.pulseIndex))
        self.TempABStart=np.zeros(len(self.pulseIndex))
        self.TempABEnd=np.zeros(len(self.pulseIndex))
        self.TempDifferenceABs=np.zeros(len(self.pulseIndex))
        c = 0
        for i in range( 1, len( self.pulserawData2[5] ) - 1 ) :
            if self.pulserawData2[5][i - 1] - self.pulserawData2[5][i] > tol and c < len(self.pulseIndex):
                self.TempABStart[c] = self.pulserawData2[5][i - 1]
                self.TempABEnd[c] = self.pulserawData2[5][i + 1]
                self.TempDifferenceABs[c] = self.pulserawData2[5][i - 1] - self.pulserawData2[5][i + 1]
                self.TimeABs[c] = self.pulserawData2[0][i]
                print('Pulse ' + str(c + 1) + ' Done')
                c += 1
        
        list1=[]
        for j in range( -1, len(self.pulseIndex) ):
            if j == -1:
                list1.append("{0}\t{1}\t{2}\t{3}\n".format('#[ 1 ] T for AB start (mK)','#[ 2 ] T for AB end (mK)', '#[ 3 ] Delta T for AB (mK)', '#[ 4 ] Time of AB') )
            else:
                list1.append("{0}\t{1}\t{2}\t{3}\n".format(self.TempABStart[j], self.TempABEnd[j], self.TempDifferenceABs[j], self.TimeABs[j]))
                
        str1 = ''.join(list1)
        self.avgTemp = round( sum(self.pulserawData2[2]) / len(self.pulserawData2[2]) , 2 )
        path1 = self.impDir + 'Python Analysis\\Pulsing_AB\\' + str(self.Ppsi) +' psi ' + str(self.avgTemp) + ' (mK) Pulses.dat'
        with open(path1,'w') as file1:
            file1.write(str1)
        
        
#%% Choose a
a = 3

#%% 6/02 
if a == 1:
    end = 1000000
    P1 = pulse(306,-1, 1, 7540, 9400, 12210, 27080, 10, 100, 602)
    #  7540 to 9400
    #  12210 to 27080 for pulsing
    # psi, cooling (-1) or warming (1) for non-pulsing (base) data, start and stop for base data
    # start and stop for pulsing data, number of points to remove before pulse, after pulse, start date
    
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed
    
    del end
    
    if P1.doneCutting == 1  :
    
        P1.temp_fit(1)
        print(P1.T_fit)
        P1.QtotimeF1(11)
        print(P1.fit_QtoF1)
        P1.reCallibrateF2(638)
    #        638
        P1.k_NNsearch(0.05)
        P1.tempAdjustPulses()
        finalPlot(P1)
        P1.findABs(0.007)
        
#%% 6/02 

if a == 2:
    end = 1000000
    P2 = pulse(306,-1, 1, 7540, 9400, 30852, 44571, 10, 100, 602)
    #  7540 to 9400
    #  
    # psi, cooling (-1) or warming (1) for non-pulsing (base) data, start and stop for base data
    # start and stop for pulsing data, number of points to remove before pulse, after pulse, start date
    
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed
    
    del end
    
    if P2.doneCutting == 1  :
    
        P2.temp_fit(1)
        print(P2.T_fit)
        P2.QtotimeF1(11)
        print(P2.fit_QtoF1)
        P2.reCallibrateF2(638)
    #        638
        P2.k_NNsearch(0.05)
        P2.tempAdjustPulses()
        finalPlot(P2)
        P2.findABs(0.007)
        
