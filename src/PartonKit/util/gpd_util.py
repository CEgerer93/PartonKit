#!/usr/bin/python

import numpy as np
import scipy.special as spec
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
from pitd_util import *

from pdf_utils import pitdJacobi, propError, texpKernel
import fit_util


insertions = { 3: { 'Redstar': 'b_b1xDA__J1_T1pP', 'CK': None },
               8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK': 'insertion_gt' },
               11: { 'Redstar': 'a_a1xDA__J1_T1pM', 'CK': None} }


class doubleDistribution():
    def __init__(self,beta,alpha,profileWidth):
        self.beta=beta
        self.alpha=alpha
        self.b=profileWidth

    def profile(self):
        return ( (1.0*spec.gamma(2*self.b+2))/(np.power(2,2*self.b+1)\
                                               *np.power(spec.gamma(self.b+1),2)) )\
                                               *(np.power((1-np.abs(self.beta))**2\
                                                          -self.alpha**2,self.b)/\
                                                 np.power(1-np.abs(self.beta),2*self.b+1))



class pgitdJacobi(pitdJacobi):
    nu=np.linspace(0,15,250)
    x=np.linspace(0,1,300)

    def __init__(self,name,colors,fitParamsAll,pNums,correlated,cfgs,comp,corrStart,dirac=8):
        self.name=name
        self.cols=colors
        self.fitParamsAll=fitParamsAll
        self.pNums=pNums # num params for lt,az,{t4,t6,...}
        self.correlated=correlated
        self.cfgs=cfgs
        self.comp=comp
        self.pAvg={}
        self.pCov={}
        self.pCovView=None # Array-like to view parameter covariance
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.kernel='J'
        self.cs=corrStart # order at which corrections start
        self.dirac=dirac # dirac structure of insertion
        self.variety='f' if self.dirac == 8 else 'g'

        self.numLT=self.pNums[0]
        self.numAZ=self.pNums[1]
        self.numHTs=[h for h in self.pNums[2:]]
        self.numParams=2+self.numLT+self.numAZ+sum(self.numHTs)

        # Initialize all kernels + derivs used in jacobi pgitd fits
        self.texpKernel=texpKernel(self.comp)
        # Change kernels to be tree-level
        self.texpKernel.NLO = self.texpKernel.Tree
        self.texpKernel.dA_NLO = self.texpKernel.dA_Tree
        self.texpKernel.dB_NLO = self.texpKernel.dB_Tree


    def lt_pgitd(self,nu):
        summ=0.0
        for n in range(0,self.numLT):
            summ+=self.pAvg['C^{lt}_'+str(n)]*\
                self.texpKernel.NLO(n,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
        return summ


    #--------------------------------------------------------
    # Leading-Twist pitd Error for some z (Continuous in \nu)
    #--------------------------------------------------------
    def lt_pgitd_error(self,nu):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for l in range(0,self.numLT):
            partials.update({'C^{lt}_%d'%l: 0.0})
            
        for n in range(0,self.numLT):
            partials['\\alpha'] += self.texpKernel.dA_NLO(n, 75, self.pAvg['\\alpha'],\
                                                          self.pAvg['\\beta'], nu)*\
                                                          self.pAvg['C^{lt}_'+str(n)]
            partials['\\beta'] += self.texpKernel.dB_NLO(n, 75, self.pAvg['\\alpha'],\
                                                         self.pAvg['\\beta'], nu)*\
                                                         self.pAvg['C^{lt}_'+str(n)]
            partials['C^{lt}_'+str(n)] += self.texpKernel.NLO(n, 75, self.pAvg['\\alpha'],\
                                                              self.pAvg['\\beta'], nu)
        # Propagate error and return
        return propError(self.pCov,partials)

    
    

    def pgitd(self,nu):
        return self.lt_pgitd(nu)

    #---------------------------------------------------------------------------------
    # The pITD Error Band predicted by model and corrections for some z (Cont. in \nu)
    #---------------------------------------------------------------------------------
    def pgitdError(self,nu):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for l in range(0,self.numLT):
            partials.update({'C^{lt}_%d'%l: 0.0})

        for n in range(0,self.numLT):
            partials['\\alpha'] += self.texpKernel.dA_NLO(n, 75, self.pAvg['\\alpha'],\
                                                          self.pAvg['\\beta'], nu)*\
                                                          self.pAvg['C^{lt}_'+str(n)]
            partials['\\beta'] += self.texpKernel.dB_NLO(n, 75, self.pAvg['\\alpha'],\
                                                         self.pAvg['\\beta'], nu)*\
                                                         self.pAvg['C^{lt}_'+str(n)]
            partials['C^{lt}_'+str(n)] += self.texpKernel.NLO(n, 75, self.pAvg['\\alpha'],\
                                                              self.pAvg['\\beta'], nu)
            
        return propError(self.pCov,partials)



        
    def plotPGITD(self,ax,col,hatch=None):
        ax.plot(self.nu,self.pgitd(self.nu),color=col,\
                label=r'$\mathfrak{%s}\ \mathfrak{M}(\nu,z^2)$'%self.comp)
        ax.fill_between(self.nu,[x+y for x, y in\
                                 zip(self.pgitd(self.nu),self.pgitdError(self.nu))],\
                        [x-y for x, y in zip(self.pgitd(self.nu),self.pgitdError(self.nu))],\
                        color=col,alpha=0.3,\
                        label=r'$\mathfrak{%s}\ \mathfrak{M}(\nu,z^2)$'%self.comp,lw=0,hatch=hatch)
        

        # # Truncation string to include with PDF/ITD labeling
        # self.truncStr="%i%i"%(self.numLT,self.numAZ)
        # for n,h in enumerate(self.numHTs):
        #     # No longer want twist-6/twist-8 pieces, so just truncate the enumeration to keep functionality
        #     if n < 2:
        #         self.truncStr+="%i"%h
        
        # # Initialize all kernels + derivs used in jacobi pitd fits
        # self.texpKernel=texpKernel(self.comp)




class h5Plot():
    def __init__(self,h5s,dtypeName,h52pt,cfgs,pf,pi,zmin,zmax,Lx,dirac=8):
        self.h5s=h5s
        self.dtypeName=dtypeName
        self.h52pt=h52pt
        self.cfgs=cfgs
        self.pf=pf
        self.pi=pi
        self.zmin=zmin
        self.zmax=zmax
        self.Lx=Lx
        self.dirac=dirac
        self.Ef=0.0
        self.Ei=0.0
        self.m=0.0
        self.compStr={'real': 'Re', 'imag': 'Im'}
        self.ampDict={} # dictionary relating axes to components of M & L amplitudes
        self.symbs=['o','^','s']
        self.STEP=(abs(ioffeTime(pi,'0.0.1',self.Lx))+abs(ioffeTime(pf,'0.0.1',self.Lx)))/2



    ##########################################
    # CREATE FIGURES & SET INITIAL AXES LABELS
    ##########################################
    def initAxes(self,formAxes=True):
        for amp in ['A%i'%i for i in range(1,11)]:
            self.ampDict.update({amp: { 'real': { 'fig': None, 'ax': None},
                                        'imag': { 'fig': None, 'ax': None}}})
        if formAxes:
            for k,v in self.ampDict.items():
                for comp in self.compStr.keys():
                    v[comp]['fig']=plt.figure(figsize=(11,8))
                    v[comp]['ax']=v[comp]['fig'].gca()
                    v[comp]['ax'].set_xlabel(r'$\nu\equiv\left(\nu_f+\nu_i\right)/2$')
                    v[comp]['ax'].\
                        set_ylabel(r'$\mathfrak{%s}\ \mathfrak{%s}\left(\nu,\xi,t;z^2\right)$'\
                                   %(self.compStr[comp],k))

        
    ####################################
    # OPEN 2PT FIT H5 TO ACCESS Ef,Ei,m
    ####################################
    def read2pts(self):
        h2pt=h5py.File(self.h52pt,'r')

        tSeries={'000': None, toStr(self.pf): None, toStr(self.pi): None}
        for k in tSeries.keys():
            tmin2pt, tmax2pt = list(h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%k].\
                                    keys())[0].replace('tfit_','').split('-')
            tSeries[k]='%s.1.%s'%(tmin2pt,tmax2pt)
    
        rest2pt=fit_util.twoState(h2pt,self.cfgs,'000','000','real',\
                                  tfitSeries=tSeries['000'],col='k')
        pi2pt=fit_util.twoState(h2pt,self.cfgs,toStr(self.pi),toStr(self.pi),\
                                'real',tfitSeries=tSeries[toStr(self.pi)],col='k')
        pf2pt=fit_util.twoState(h2pt,self.cfgs,toStr(self.pf),toStr(self.pf),\
                                'real',tfitSeries=tSeries[toStr(self.pf)],col='k')

        for c in [rest2pt, pi2pt, pf2pt]: c.parse()
        h2pt.close()
        
        # Set average Ef,Ei,m for momentum transfer evaluation
        self.Ef=pf2pt.pAvg['E0']
        self.Ei=pi2pt.pAvg['E0']
        self.m=rest2pt.pAvg['E0']


    ####################################
    # ACCESS FILE HANDLE(S)
    ####################################
    def showPGITDData(self,useAlternateAxis=None,symbOverride=None):
        for nH5, h5file in enumerate(self.h5s.split(':')):
            h5In = h5py.File(h5file,'r')
            dtypeName = self.dtypeName.split(':')[nH5]

            for amp in self.ampDict.keys():
                for z in range(self.zmin,self.zmax+1):
                    ztag="zsep00%d"%z
                    ptag="pf%s_pi%s"%(toStr(self.pf),toStr(self.pi))
                    for comp in self.compStr.keys():
                        ioffeTime = -1; avgMat = 0.0; avgMatErr = 0.0

                        # Guard for non-valid entry
                        try:
                            # Access h5 once
                            mats=h5In['/%s/%s/%s/%s/jack/%s/%s'%\
                                      (insertions[self.dirac]['Redstar'],\
                                       amp,ztag,ptag,comp,dtypeName)]
                            
                            # Get avgMat & Ioffe Time
                            ioffeTime, avgMat = sum(mats)
                            avgMat *= (1.0/self.cfgs)
                            ioffeTime *= (1.0/self.cfgs)
                            
                            # Get avgMat's error
                            avgMatErr = sum(np.power(mat-avgMat,2) for mat in mats[:,1])
                            avgMatErr = np.sqrt( ((1.0*(self.cfgs-1))/self.cfgs)*avgMatErr )
                            
                            
                            # Default to plot on this h5Plot's ax instance, but potentially accept another
                            ax=self.ampDict[amp][comp]['ax'] if not useAlternateAxis\
                                else useAlternateAxis[amp][comp]['ax']
                            ax.errorbar(ioffeTime+nH5*0.1*self.STEP,\
                                        avgMat, yerr=avgMatErr,\
                                        fmt=self.symbs[nH5] if not symbOverride else symbOverride,\
                                        color=mainColors[z],\
                                        mec=mainColors[z],\
                                        mfc=mainColors[z],label=r'$z=%s$'%z)
                        except:
                            raise Exception("---> Non-valid path: /%s/%s/%s/%s/jack/%s/%s"\
                                            %(insertions[self.dirac]['Redstar'],\
                                              amp,ztag,ptag,comp,dtypeName))
                            

###################################################################################################
# Dictionary containing operator names used in GPD 3pt contractions and associated 2pt contractions
###################################################################################################
# gpdOpDict={'000': 'Nucleon_proj0_p000_G1g',\
#            '001': 'Nucleon_proj0_p100_H1o2D4E1',\
#            '00-1': 'Nucleon_proj1_p100_H1o2D4E1',\
#            '002': 'Nucleon_proj0_p200_H1o2D4E1',\
#            '00-3': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
#            '01-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '011': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '01-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '012': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '10-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '101': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '10-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '102': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '11-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D3E1',\
#            '111': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D3E1',\
#            '11-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '112': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '11-3': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '20-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '201': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
#            '20-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '202': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
#            '21-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '211': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '21-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
#            '212': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE'}


gpdOpDict={'000': 'NucleonMG1g1MxD0J0S_J1o2_G1g1',\
           '001': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '010': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '100': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '00-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '002': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '00-3': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D4E1',\
           '01-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '011': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '01-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '012': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '10-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '101': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '10-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '102': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '11-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D3E1',\
           '111': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D3E1',\
           '11-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '112': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '11-3': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '20-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '201': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nm0E',\
           '20-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '202': 'NucleonMG1g1MxD0J0S_J1o2_H1o2D2E',\
           '21-1': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '211': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '21-2': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE',\
           '212': 'NucleonMG1g1MxD0J0S_J1o2_H1o2C4nnmE'}
