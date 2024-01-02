#!/usr/bin/python3
import numpy as np
import h5py
import sys
sys.path.append("/home/colin/QCD/pseudoDists/strFuncViz")
from gpd_utils import *
from common_util import *

class common:
    # Make jackknife samples for an input array
    def jk(self):
        dat=self.dat
        cfgs=self.cfgs
        Nt=self.Nt
        self.jks=np.zeros((cfgs,cfgs-1,Nt))
        for n in range(0,cfgs):
            print("Making jackknife #%i"%n)
            if n > 0:
                for l in range(0,n):
                    self.jks[n,l,:]=dat[l][:]
                    # for t in range(0,Nt):
                    #     self.jks[n,l,t]+=dat[l][t]
                    
            for l in range(n,cfgs-1):
                self.jks[n,l,:]=dat[l+1][:]
                # for t in range(0,Nt):
                #     self.jks[n,l,t]+=dat[l+1][t]


    def makeAvg(self):
        self.avg=np.zeros(self.Nt)
        for t in range(0,self.Nt):
            for g in range(0,self.cfgs):
                self.avg[t]+=self.dat[g][t]
            self.avg[t] /= (1.0*self.cfgs)

    def makeCov(self):
        self.cov=np.zeros((self.Nt,self.Nt))
        for ti in range(0,self.Nt):
            for tj in range(0,self.Nt):
                for g in range(0,self.cfgs):
                    self.cov[ti][tj]+=(self.jkAvg[g][ti]-self.avg[ti])*\
                        (self.jkAvg[g][tj]-self.avg[tj])
                self.cov[ti][tj] *= ((1.0*(self.cfgs-1))/self.cfgs)

    def avgJks(self):
        self.jkAvg=np.zeros((self.cfgs,self.Nt))
        # Form an average per jackknife sample
        for n in range(0,self.cfgs):
            for t in range(0,self.Nt):
                dum=0.0
                for g in range(0,self.cfgs-1):
                    dum+=self.jks[n,g,t]
                        
                self.jkAvg[n,t]=(1.0/(1.0*(self.cfgs-1)))*dum

class corr2pt(common):
    def __init__(self,datH5,cfgs,tSeries,_p,comp,col,StoN=0.0,rowSrc=1,rowSnk=1,gpdOps=False):
        self.datH5=datH5
        self.cfgs=cfgs
        self.tmin, self.tstep, self.tmax = tuple(int(t) for t in tSeries.split('.'))
        self.T=np.linspace(self.tmin,self.tmax,\
                           int((self.tmax-self.tmin+self.tstep)/self.tstep))
        self.Nt=len(self.T)
        self.p=_p
        self.comp=comp
        self.rs=rowSrc
        self.rk=rowSnk
        self.col=col
        self.StoN=StoN
        self.dat=None
        self.avg=None
        self.jks=None
        self.jkAvg=None
        self.cov=None
        self.ops={}

        # If gpdOps is False, then use PDF operators
        if not gpdOps:
            for p in range(-6,7):
                if p == 0:
                    self.ops.update({'00%i'%p: 'NucleonMG1g1MxD0J0S_J1o2_G1g1'})
                else:
                    self.ops.update({'00%i'%p: 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1'})
        else:
            self.ops=gpdOpDict

            
                    
    def readDat(self,typ='h5'):
        if typ != 'h5':
            raise ValueError("Can only read data in h5 format!")
        else:
            try:
                self.dat = self.datH5['/p%s/t0_avg/%s-%s/rows_%i%i/%s/data'\
                                      %(self.p,self.ops[self.p],self.ops[self.p],\
                                        self.rk,self.rs,self.comp)]
            except:
                raise ValueError("/p%s/t0_avg/%s-%s/rows_%i%i/%s/data  not found"%\
                                 (self.p,self.ops[self.p],self.ops[self.p],\
                                  self.rk,self.rs,self.comp))
            
    # Plot correlator
    def plot(self,ax,eff):
        err=[]
        # Standard
        if not eff:
            for n,v in enumerate(self.T):
                err.append(np.sqrt(self.cov[n][n]))
            ax.errorbar(self.T,self.avg,yerr=err,fmt='o',color=self.col,mfc=self.col,\
                        mec=self.col,capsize=3,markersize=3,lw=1)
        # Effective correlator - i.e. effective energy
        else:
            effT=np.linspace( (self.tmin+self.tstep/2.0),\
                                   (self.tmax-self.tstep/2.0), self.Nt-1)
            err=np.zeros(self.Nt-1)
            eff=np.zeros((self.cfgs,self.Nt-1))
            for t in range(0,self.Nt-1):
                eff[:,t]=(1.0/self.tstep)*np.log(self.jkAvg[:,t]/self.jkAvg[:,t+1])
                
            effAvg=np.sum(eff,axis=0) # reduction
            effAvg[:]/=self.cfgs

            for t in range(0,self.Nt-1):
                dum=0.0
                for j in range(0,self.cfgs):
                    dum+=np.power(eff[j,t]-effAvg[t],2)

                err[t]=np.sqrt(((self.cfgs-1)/(1.0*self.cfgs))*dum)

            ###### How to plot the error
            # Plot all the data, if StoN is default (0.0)
            if self.StoN == 0.0:
                ax.errorbar(effT,effAvg,yerr=err,fmt='o',color=self.col,mfc=self.col,\
                            mec=self.col,capsize=3,markersize=3,lw=1,\
                            label=r'$E^{\rm eff}_0\left(%s,%s,%s\right)$'%as_tuple(self.p))
            else:
                for n, v in enumerate(effAvg):
                    if v/err[n] > self.StoN:
                        ax.errorbar(effT[n],v,yerr=err[n],fmt='o',color=self.col,mfc=self.col,\
                                    mec=self.col,capsize=3,markersize=3,lw=1,\
                                    label=r'$E^{\rm eff}_0\left(%s,%s,%s\right)$'%as_tuple(self.p))
            

class corrSR(common):
    # Constructor when given #cfgs and (Cfg x Nt) data array
    def __init__(self,datH5,cfgs,Nt,pf,pi,rowf,rowi,comp,z=None,gam=None):
        self.datH5=datH5
        self.cfgs=cfgs
        self.Nt=Nt
        self.pf=pf
        self.pi=pi
        self.rowf=rowf
        self.rowi=rowi
        self.comp=comp
        self.z=z
        self.gam=gam

        self.dat=None
        
        self.avg=None
        self.jks=None
        self.jkAvg=None
        self.cov=None

        self.ops={}

        # PDF operators
        if pf == pi:
            for p in range(-6,7):
                if p == 0:
                    self.ops.update({'00%i'%p: 'NucleonMG1g1MxD0J0S_J1o2_G1g1'})
                else:
                    self.ops.update({'00%i'%p: 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1'})
        # GPD operators
        else:
            self.ops = gpdOpDict
                            

    # Read summed ratio data
    def readDat(self,typ='h5'):
        if typ != 'h5':
            raise ValueError("Can only read data in h5 format!")
            sys.exit()
        else:
            h = h5py.File(self.datH5,'r')

            searchStr='/pf%s_pi%s/zsep%s/gamma-%i/t0_avg/%s-%s/rows_%s%s/%s/data'\
                %(self.pf,self.pi,self.z,self.gam,self.ops[self.pf],\
                  self.ops[self.pi],self.rowf,self.rowi,self.comp)
            if self.z == None:
                searchStr='/pf%s_pi%s/gamma-%i/t0_avg/%s-%s/rows_%s%s/%s/data'\
                    %(self.pf,self.pi,self.gam,self.ops[self.pf],\
                      self.ops[self.pi],self.rowf,self.rowi,self.comp)

            try: # rows -1 -1 for PDF 
                self.dat = h[searchStr]
            except:
                raise ValueError(searchStr)
                sys.exit()
