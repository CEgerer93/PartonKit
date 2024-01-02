#!/usr/bin/python
################
# Some shared functions to import
################

import numpy as np

def jackknifeSamples(data,complexity,numT,gauge_configs):
    dataJks=np.zeros((gauge_configs,gauge_configs-1,numT))
    # Make jackknife samples for an input array
    for n in range(0,gauge_configs):
        if n > 0:
            for l in range(0,n):
                for t in range(0,numT):
                    dataJks[n,l,t]=data[t+l*numT,complexity]
                    
        for l in range(n,gauge_configs-1):
            for t in range(0,numT):
                dataJks[n,l,t]=data[t+(l+1)*numT,complexity]

    return dataJks


def avgJackknifeSamples(data,numT,gauge_configs):
    dataJkAvg=np.zeros((gauge_configs,numT))
    # Form an average per jackknife sample
    for n in range(0,gauge_configs):
        for t in range(0,numT):
            dum=0.0
            for g in range(0,gauge_configs-1):
                dum+=data[n,g,t]

            dataJkAvg[n,t]=(1.0/(1.0*(gauge_configs-1)))*dum

    return dataJkAvg


# # Make jackknife samples
# def makeJks(d):
#     jks=np.zeros((len(d),len(d)-1))
#     for j in range(0,len(d)):
#         if j > 0:
#             for l in range(0,j):
#                 jks[j][l]=d[l]
#         for l in range(j,len(d)-1):
#             jks[j][l]=d[l+1]
#     return jks

# # Average within each jackknife sample
# def makeAvgJks(jks):
#     avgJks = np.zeros(len(jks))
#     for g in range(0,len(jks)):
#         dum=0.0
#         for gi in range(0,len(jks)-1):
#             dum+=jks[g][gi]
#         avgJks[g] = (1.0/(1.0*(len(jks)-1)))*dum
#     return avgJks


############################################################################
# A generic class to contain 1D data arrays & their jackknife samples/avgs
############################################################################
class generic1D:
    def __init__(self,dat=[],avg=None,jks=None,jkAvg=None,cov=None):
        self.dat=list(dat)
        self.avg=avg
        self.jks=jks
        self.jkAvg=jkAvg
        self.cov=cov
        self.dim=len(self.dat)

    def average(self):
        self.avg=0.0
        for j in range(0,self.dim):
            self.avg+=self.dat[j]
        self.avg *= (1.0/self.dim)

        return self.avg

    def makeJks(self):
        self.jks=np.zeros((self.dim,self.dim-1))
        for j in range(0,self.dim):
            if j > 0:
                for l in range(0,j):
                    self.jks[j][l]=self.dat[l]
            for l in range(j,self.dim-1):
                self.jks[j][l]=self.dat[l+1]

    def makeAvgJks(self):
        self.jkAvg = np.zeros(len(self.jks))
        for g in range(0,len(self.jks)):
            dum=0.0
            for gi in range(0,len(self.jks)-1):
                dum+=self.jks[g][gi]
            self.jkAvg[g] = (1.0/(1.0*(len(self.jks)-1)))*dum

    def removeBias(self):
        dum = self.dat # local copy
        for j in range(0,self.dim):
            dum[j] = self.dim*self.avg - (self.dim-1)*self.dat[j]
        # Remap dum to self.dat - now bias is corrected
        self.dat = dum
        
