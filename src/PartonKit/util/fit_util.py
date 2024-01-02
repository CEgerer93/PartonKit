#!/usr/bin/python3
import numpy as np
import h5py

import time

class utils:
    def parse(self,parseOnly=None):
        # Parameters to consider - all by default
        # paramsToParse should always be cast as a dict
        paramsToParse = self.params if parseOnly is None else {parseOnly : self.params[parseOnly]}

        # Form the ensemble avg of each fit parameter, by dividing by number of gauge configs
        for k, v in paramsToParse.items():
            # Sum all params, skipping 'norm' if it is None
            dum=0.0
            if v is None:
                continue
            else:
                dum=sum(v)/(1.0*self.cfgs)
            self.pAvg.update({k: dum})

        t0=time.time()
        # Form the parameter covariance
        for ki in self.pAvg.keys():
            for kj in self.pAvg.keys():
                key=(ki,kj)
                dum=0.0

                tg_0=time.time()
                
                dum=sum([(i-self.pAvg[ki])*(j-self.pAvg[kj]) for i,j in zip(self.params[ki],self.params[kj])])

                dum*=((1.0*(self.cfgs-1))/self.cfgs)

                tg_1=time.time()
                print("  Inner time = %.5f"%(tg_1-tg_0))

                if not(self.correlated) and ki != kj:
                    self.pCov.update({key: 0.0})
                else:
                    self.pCov.update({key: dum})

        # Init partials
        for k in self.pAvg.keys():
            self.partials.update({k: 0.0})
        t1=time.time()
        print("Total parse time = %.5f"%(t1-t0))
        

    def printFitParams(self):
        for k in self.params.keys():
            print("%s = %.7f +/- %.7f"%(k,self.pAvg[k],np.sqrt(self.pCov[(k,k)])))

            
    # Plot the fit
    def plot(self,ax,eff,fitTMin,fitTMax):
        theFit={'avg': None, 'err': None}

        if not eff:
            theFit['avg']=[self.func(T) for T in self.tFit]
            theFit['err']=[self.err(T) for T in self.tFit]

            A=ax.plot(self.tFit,theFit['avg'],color=self.col)
            B=ax.fill_between(self.tFit,[x-y for x,y in zip(theFit['avg'],theFit['err'])],
                              [x+y for x,y in zip(theFit['avg'],theFit['err'])],
                              color=self.col,alpha=0.5,label=self.label,lw=0)

        else:
            tEff=np.linspace(fitTMin-2,fitTMax+2,self.Nt-1)

            theFit['avg']=np.zeros(self.Nt-1)
            theFit['err']=np.zeros(self.Nt-1)

            
            for n in range(0,self.Nt-2):
                theFit['avg'][n]=(1.0/(tEff[1]-tEff[0]))*\
                    np.log(self.func(tEff[n])/self.func(tEff[n+1]))

                theFit['err'][n]=self.effErr(tEff[n],tEff[1]-tEff[0])

            # Plot all except last point
            # ax.plot(tEff[:-1],theFit['avg'][:-1],color=self.col,lw=1)
            # tminIdx=np.find_nearest(tEff[:-1],self.tmin)
            # tmaxIdx=np.find_nearest(tEff[:-1],self.tmax)
            # ax.plot(theFit['avg'][n],color=self.col,lw=10)
                    
                
            ax.fill_between(tEff[:-1],[x-y for x,y in zip(theFit['avg'],theFit['err'])][:-1],
                            [x+y for x,y in zip(theFit['avg'],theFit['err'])][:-1],
                            color=self.col,alpha=0.35,label=self.label,lw=0,\
                            where=np.logical_and(tEff[:-1]>=self.tmin,tEff[:-1]<=self.tmax))
            
            ax.fill_between(tEff[:-1],[x-y for x,y in zip(theFit['avg'],theFit['err'])][:-1],
                            [x+y for x,y in zip(theFit['avg'],theFit['err'])][:-1],
                            color=self.col,alpha=0.12, lw=0,\
                            where=np.logical_or(tEff[:-1]<=self.tmin,tEff[:-1]>=self.tmax))

###########################
# Polarization Vector
###########################
class polVec(utils):
    def __init__(self,h5,cfgs,pf,pi,idx,comp):
        self.h=h5py.File(h5,'r')
        self.cfgs=cfgs
        self.pf=pf
        self.pi=pi
        self.idx=idx
        self.comp=comp

        self.pAvg={}
        self.pCov={}
        self.correlated=True

        self.params={}
        self.params.update({'S^mu': self.h['/polVec/%i/bins/%s/%s-%s'%
                                           (self.idx,self.comp,self.pf,self.pi)]})
        
class SR(utils):
    def __init__(self,h5,cfgs,pf,pi,rowf,rowi,z,gam,comp,tmin,tmax,fitFunc='a+bT',pdfH5=False):
        self.h=h5py.File(h5,'r')
        self.cfgs=cfgs
        self.pf=pf
        self.pi=pi
        self.rowf=rowf
        self.rowi=rowi
        self.z=z
        self.gam=gam
        self.comp=comp
        self.tmin=tmin
        self.tmax=tmax
        self.fitFunc=fitFunc
        self.pdfH5=pdfH5
        self.label=r'${\rm Fit}\ T/a\in\left[%s,%s\right]$'%(self.tmin,self.tmax)

        self.pAvg={}
        self.pCov={}
        self.correlated=True

        self.params={}
        self.paramOrder={}
        # self.paramOrder = {0: 'a', 1: 'b', 2: 'chi2'}
        self.partials = {}

        cnt=0
        # Determine all distinct param types
        for k,v in self.h['/%s'%self.fitFunc].items():
            self.paramOrder.update({cnt: k})
            if self.pdfH5:
                try:
                    self.params.update({k: v['bins/%s/%s_%s/%s/gamma-%i/tfit_%i-%i'\
                                             %(self.comp,self.pf,self.pi,self.z,self.gam,tmin,tmax)]})
                except:
                    raise Exception("---> Cannot access directory = bins/%s/%s_%s/%s/gamma-%i/tfit_%i-%i"\
                                             %(self.comp,self.pf,self.pi,self.z,self.gam,tmin,tmax))
            else:
                if self.z == None:
                    try:
                        self.params.update({k:\
                                            v['bins/%s/%s_%s/rowf%s_rowi%s/gamma-%i/tfit_%i-%i'\
                                              %(self.comp,self.pf,self.pi,self.rowf,self.rowi,\
                                                self.gam,tmin,tmax)]})
                    except:
                        raise Exception("---> Cannot access directory = bins/%s/%s_%s/rowf%s_rowi%s/gamma-%i/tfit_%i-%i"\
                                              %(self.comp,self.pf,self.pi,self.rowf,self.rowi,\
                                                self.gam,tmin,tmax))
                else:
                    try:
                        self.params.update({k:\
                                            v['bins/%s/%s_%s/rowf%s_rowi%s/%s/gamma-%i/tfit_%i-%i'\
                                              %(self.comp,self.pf,self.pi,self.rowf,self.rowi,\
                                                self.z,self.gam,tmin,tmax)]})
                    except:
                        raise Exception("---> Cannot access directory = bins/%s/%s_%s/rowf%s_rowi%s/%s/gamma-%i/tfit_%i-%i"\
                                        %(self.comp,self.pf,self.pi,self.rowf,self.rowi,\
                                          self.z,self.gam,tmin,tmax))
                        

    # Central value of fit
    def func(self,T):
        return self.pAvg['a']+self.pAvg['b']*T


    # Correlated error of fit
    def err(self,T):
        self.partials['a']=1
        self.partials['b']=T

        t0=time.time()
        error=0.0

        for i in self.pAvg:
            error+=self.partials[i]*sum(self.pCov[(i, j)]*self.partials[j] for j in self.pAvg)

        tf=time.time()
        print("SR Err time = %.8f"%(tf-t0))
        return np.sqrt(error)



##########################
# 2-STATE FIT
##########################
class twoState(utils):
    def __init__(self,h5,cfgs,pf,pi,comp,tfitSeries,col,z=None,gam=None,\
                 fitFunc='exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}'):
        # self.h=h5py.File(h5,'r')
        self.h=h5
        self.cfgs=cfgs
        self.pf=pf
        self.pi=pi
        self.p=None
        if self.pf == self.pi: self.p=self.pi
        self.comp=comp
        self.tmin, self.tstep, self.tmax = tuple(int(ti) for ti in tfitSeries.split('.'))
        self.tFit=np.linspace(self.tmin-0.5*self.tstep,self.tmax+0.5*self.tstep,3000)
        self.Nt=len(self.tFit)
        self.col=col
        self.z=z
        self.gam=gam
        self.fitFunc=fitFunc
        self.label=''

        self.pAvg={}
        self.pCov={}
        self.correlated=True

        self.params={}
        self.paramOrder={}
        self.partials={}

        cnt=0
        # Determine all distinct param types
        for k,v in self.h['/%s'%self.fitFunc].items():
            self.paramOrder.update({cnt: k})
            if self.p is not None:
                hierarchy='bins/%s/p%s/tfit_%i-%i'\
                    %(self.comp,self.p,self.tmin,self.tmax)
            elif self.p is not None and self.z is not None and self.gam is not None:
                hierarchy='bins/%s/p%s/zsep%s/gamma-%i/tfit_%i-%i'\
                    %(self.comp,self.p,self.z,self.gam,self.tmin,self.tmax)
            else:
                hierarchy='bins/%s/pf%s_pi%s/zsep%s/gamma-%i/tfit_%i-%i'\
                    %(self.comp,self.pf,self.pi,self.z,self.gam,self.tmin,self.tmax)
            try:
                self.params.update({k: v[hierarchy]})
            except:
                raise Exception("Could not open hierarchy = %s"%hierarchy)

            
    # Central value of fit
    def func(self,T):
        return np.exp(-self.pAvg['E0']*T)*(self.pAvg['a']+self.pAvg['b']*np.exp(-(self.pAvg['E1']-self.pAvg['E0'])*T))

    # Correlated error of fit
    def err(self,T):
        self.partials['a']=np.exp(-self.pAvg['E0']*T)
        self.partials['b']=np.exp(-self.pAvg['E1']*T)
        self.partials['E0']=-self.pAvg['a']*np.exp(-self.pAvg['E0']*T)*T
        self.partials['E1']=-self.pAvg['b']*np.exp(-self.pAvg['E1']*T)*T

        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=self.partials[i]*self.pCov[(i, j)]*self.partials[j]
        return np.sqrt(error)


    # Correlated error of effective fit
    def effErr(self,T,tdiff):
        self.partials['a']=(1.0/tdiff)*(1/(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*T))-1/(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+T))))
        self.partials['b']=(self.pAvg['a']/(self.pAvg['b']*tdiff))*(-(1/(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*T)))+1/(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+T))))
        self.partials['E0']=(self.pAvg['a']*(self.pAvg['a']*tdiff+self.pAvg['b']*np.exp(self.pAvg['E0']*T-self.pAvg['E1']*(tdiff+T))*(-np.exp(tdiff*self.pAvg['E0'])*T+np.exp(tdiff*self.pAvg['E1'])*(tdiff+T))))/(tdiff*(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*T))*(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+T))))
        self.partials['E1']=(self.pAvg['b']*(self.pAvg['b']*tdiff*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+2*T))-self.pAvg['a']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*T)*T+self.pAvg['a']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+T))*(tdiff+T)))/(tdiff*(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*T))*(self.pAvg['a']+self.pAvg['b']*np.exp((self.pAvg['E0']-self.pAvg['E1'])*(tdiff+T))))

        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=self.partials[i]*self.pCov[(i, j)]*self.partials[j]
        return np.sqrt(error)


    # Overlaps
    def overlaps(self):
        return self.pAvg['a']*2*self.pAvg['E0'], self.pAvg['b']*2*self.pAvg['E1']
