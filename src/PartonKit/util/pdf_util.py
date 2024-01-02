#/usr/bin/python3

import h5py
import numpy as np
import scipy.special as spec
import scipy.integrate as integrate
import matplotlib.pyplot as plt
#####################
# FUNCTIONALITY FOR
#####################

insertions = { 3: { 'Redstar': 'b_b1xDA__J1_T1pP', 'CK': None },
               8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK': 'insertion_gt' },
               11: { 'Redstar': 'a_a1xDA__J1_T1pM', 'CK': None} }


# Momentum Fractions for all PDFs
xMin=0.0
xMax=1.0

# Constants
Cf=4.0/3.0
hbarc = 0.1973269804
aLat = 0.094
muRenorm = 2.0
MU = (aLat*muRenorm)/hbarc
# alphaS = 0.303
alphaS=None
LambdaQCD=0.286*(aLat/hbarc)
Lx = 32

# Harmonic number evaluation
def H(n):
    return np.euler_gamma+spec.digamma(n+1)

# Helper function for propagation of correlated error
def propError(pCov,d):
    error = 0.0
    for ki, vi in d.items():
        for kj, vj in d.items():
            error += vi*pCov[(ki,kj)]*vj
    return np.sqrt(error)

#-----------------------------------------------------------------------------------------------------------
# Jacobi Coefficients
# w_{n,j}^{(a,b)} = (-1)^j/n! * (n_C_j) * { Gamma(a+n+1)*Gamma(a+b+n+j+1) }/{ Gamma(a+j+1)*Gamma(a+b+n+1) }
#-----------------------------------------------------------------------------------------------------------
def coeffJacobi(n,j,a,b):
    return (np.power(-1,j)/(1.0*spec.factorial(n)))*spec.binom(n,j)\
        *( (1.0*spec.gamma(a+n+1)*spec.gamma(a+b+n+j+1))/(spec.gamma(a+j+1)*spec.gamma(a+b+n+1)) )

def coeffJacobi_dA(n,j,a,b):
    return (np.power(-1,1+j)*spec.binom(n,j)*spec.gamma(1+a+n)*\
            spec.gamma(1+a+b+j+n)*(H(a+j)-H(a+n)+H(a+b+n)-H(a+b+j+n)))/\
            (1.0*spec.factorial(n)*spec.gamma(1+a+j)*spec.gamma(1+a+b+n))

def coeffJacobi_dB(n,j,a,b):
    return (np.power(-1,1+j)*spec.binom(n,j)*spec.gamma(1+a+n)*\
            spec.gamma(1+a+b+j+n)*(H(a+b+n)-H(a+b+j+n)))/\
            (1.0*spec.factorial(n)*spec.gamma(1+a+j)*spec.gamma(1+a+b+n))

#--------------------------------------------------------------------------------
# Jacobi Polynomial: \omega_n^{(a,b)}(x) = \sum_{j=0}^n coeffJacobi(n,j,a,b)*x^j
#--------------------------------------------------------------------------------
def Jacobi(n,a,b,x):
    summ=0.0
    for j in range(0,n+1):
        summ += coeffJacobi(n,j,a,b)*np.power(x,j);
    return summ

#-------------------------------------------
# Derivative of Jacobi Polynomial wrt Alpha
#-------------------------------------------
def dAlphaJacobi(n,a,b,x):
    summ=0.0
    for j in range(0,n+1):
        summ += np.power(x,j)*coeffJacobi_dA(n,j,a,b)
    return summ

#------------------------------------------
# Derivative of Jacobi Polynomial wrt Beta
#------------------------------------------
def dBetaJacobi(n,a,b,x):
    summ=0.0
    for j in range(0,n+1):
        summ += np.power(x,j)*coeffJacobi_dB(n,j,a,b)
    return summ

#--------------------------------------------------
# Leading order moments of Altarelli-Parisi kernel
#--------------------------------------------------
def texp_gn(n, dirac):
    summ=0.0
    # Sum part of what would be a divergent p-series
    for k in range(2,n+2):
        summ+=1.0/k
    if dirac == 8:
        return -1.0/2 + 1.0/((n+1)*(n+2)) - 2*summ
    elif dirac == 11:
        return -1.0/2 + 1.0/((n+1)*(n+2)) - 2*summ
    else:
        raise Exception("Dirac = %i not supported!"%dirac)

#-----------------------------------
# Moments of scheme matching kernel
#-----------------------------------
def texp_dn(n, dirac):
    summ=0.0; summ2=0.0
    # Sum part of what would be a divergent p-series
    for k in range(1,n+1):
        summ += 1.0/k
        summ2 += 1.0/(k**2)
    summ = (summ**2)

    if dirac == 8:
        return 2*( summ + summ2 + 1.0/2 - 1.0/((n+1)*(n+2)) )
    elif dirac == 11:
        return 2*( summ + 1 - (2.0/((n+1)*(n+2))) + summ2 ) # If we aren't separating Y & R amps
        # return 2*( summ + summ2 + 1.0/2 - 1.0/((n+1)*(n+2)) ) # If we are separating Y from R
    else:
        raise Exception("Dirac = %i not supported!"%dirac)

#---------------------------------------------------------------
# Support for Taylor expansion of NLO pITD->PDF Matching Kernel
#---------------------------------------------------------------
def texp_cn(n, z, dirac=8):
    if dirac != 8 and dirac != 11:
        raise Exception("Dirac = %i not supported!"%dirac)
    else:
        return 1-((alphaS*Cf)/(2*np.pi))*( texp_gn(n, dirac)*np.log( (np.exp(2*np.euler_gamma+1)/4)\
                                                                     *np.power(MU*z,2) ) + texp_dn(n, dirac) )

#-------------------------------------------------------------------------------------
# Support for Taylor expansion of Real(pITD->PDF Kernel)*(Jacobi PDF Parametrization)
#-------------------------------------------------------------------------------------
def pitd_texp_sigma_n(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*texp_cn(2*k,z,dirac)*\
                coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)*np.power(nu,2*k)
    return summ
#+++++++++++++++++++++++
def pitd_texp_sigma_n_dA(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*texp_cn(2*k,z,dirac)*\
                    (coeffJacobi_dA(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)+\
                     coeffJacobi(n,j,a,b)*(spec.beta(1+a+j+2*k,1+b)*\
                                           (H(a+j+2*k)-H(1+a+b+j+2*k))))*np.power(nu,2*k)
    return summ
#+++++++++++++++++++++++
def pitd_texp_sigma_n_dB(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*texp_cn(2*k,z,dirac)*\
                    (coeffJacobi_dB(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)+\
                     coeffJacobi(n,j,a,b)*(spec.beta(1+a+j+2*k,1+b)*\
                                           (H(b)-H(1+a+b+j+2*k))))*np.power(nu,2*k)
    return summ
#=========================================================================================


#-------------------------------------------------------------------------------------
# Support for Taylor expansion of Imag(pITD->PDF Kernel)*(Jacobi PDF Parametrization)
#-------------------------------------------------------------------------------------
def pitd_texp_eta_n(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*texp_cn(2*k+1,z,dirac)*\
                    coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*np.power(nu,2*k+1)
    return summ
#+++++++++++++++++++++++++
def pitd_texp_eta_n_dA(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*texp_cn(2*k+1,z,dirac)*\
                    (coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*(H(1+a+j+2*k)-H(2+a+b+j+2*k))+\
                     coeffJacobi_dA(n,j,a,b)*spec.beta(a+2*k+j+2,b+1))*np.power(nu,2*k+1)
    return summ
#+++++++++++++++++++++++++
def pitd_texp_eta_n_dB(n, trunc, a, b, nu, z, dirac=8):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*texp_cn(2*k+1,z,dirac)*\
                    (coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*(H(b)-H(2+a+b+j+2*k))+\
                     coeffJacobi_dB(n,j,a,b)*spec.beta(a+2*k+j+2,b+1))*np.power(nu,2*k+1)
    return summ
#===========================================================================================

#--------------------------------------------------------------------------------------------------
# Support for Taylor expansion of Real(pITD->PDF Kernel *TREE LEVEL*)*(Jacobi PDF Parametrization)
#--------------------------------------------------------------------------------------------------
def pitd_texp_sigma_n_treelevel(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*1.0*\
	            coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)*np.power(nu,2*k)
    return summ
#+++++++++++++++++++++++++++++++
def pitd_texp_sigma_n_treelevel_dA(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*1.0*\
                    (coeffJacobi_dA(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)+\
                     coeffJacobi(n,j,a,b)*(spec.beta(1+a+j+2*k,1+b)*\
                                           (H(a+j+2*k)-H(1+a+b+j+2*k))))*np.power(nu,2*k)
    return summ
#+++++++++++++++++++++++++++++++
def pitd_texp_sigma_n_treelevel_dB(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k)))*1.0*\
                    (coeffJacobi_dB(n,j,a,b)*spec.beta(a+2*k+j+1,b+1)+\
                     coeffJacobi(n,j,a,b)*(spec.beta(1+a+j+2*k,1+b)*\
                                           (H(b)-H(1+a+b+j+2*k))))*np.power(nu,2*k)
    return summ
#===========================================================================================


#--------------------------------------------------------------------------------------------------
# Support for Taylor expansion of Imag(pITD->PDF Kernel *TREE LEVEL*)*(Jacobi PDF Parametrization)
#--------------------------------------------------------------------------------------------------
def pitd_texp_eta_n_treelevel(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*1.0*\
                    coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*np.power(nu,2*k+1)
    return summ
#++++++++++++++++++++++++++++++++++
def pitd_texp_eta_n_treelevel_dA(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*1.0*\
                    (coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*(H(1+a+j+2*k)-H(2+a+b+j+2*k))+\
                     coeffJacobi_dA(n,j,a,b)*spec.beta(a+2*k+j+2,b+1))*np.power(nu,2*k+1)
    return summ
#++++++++++++++++++++++++++++++++++
def pitd_texp_eta_n_treelevel_dB(n, trunc, a, b, nu):
    summ=0.0
    for j in range(0,n+1):
        for k in range(0,trunc+1):
            summ += (np.power(-1,k)/(1.0*spec.factorial(2*k+1)))*1.0*\
                    (coeffJacobi(n,j,a,b)*spec.beta(a+2*k+j+2,b+1)*(H(b)-H(2+a+b+j+2*k))+\
                     coeffJacobi_dB(n,j,a,b)*spec.beta(a+2*k+j+2,b+1))*np.power(nu,2*k+1)
    return summ
#================================================================================================


#####################################################################################
# Class to Manage kernels (& derivs wrt {a,b}) of jacobi fits to 'Re'/'Im' pitd data
#####################################################################################
class texpKernel():
    def __init__(self,comp):
        self.comp = comp
        self.NLO = self.func()[0]
        self.Tree = self.func()[1]
        self.dA_NLO = self.dfunc()[0]
        self.dB_NLO = self.dfunc()[1]
        self.dA_Tree = self.dfunc()[2]
        self.dB_Tree = self.dfunc()[3]
        self.subleadPow=0
        
        if self.comp == 'Re':
            self.subleadPow = 1 # CORRECTED! WHY WAS THIS = 2???! (07/06/2021)
        if self.comp == 'Im':
            self.subleadPow = 1
        
    def func(self):
        if self.comp == 'Re':
            NLO=pitd_texp_sigma_n
            Tree=pitd_texp_sigma_n_treelevel
        if self.comp == 'Im':
            NLO=pitd_texp_eta_n
            Tree=pitd_texp_eta_n_treelevel
        return NLO, Tree

    def dfunc(self):
        if self.comp == 'Re':
            dA_NLO=pitd_texp_sigma_n_dA
            dB_NLO=pitd_texp_sigma_n_dB
            dA_Tree=pitd_texp_sigma_n_treelevel_dA
            dB_Tree=pitd_texp_sigma_n_treelevel_dB
        if self.comp == 'Im':
            dA_NLO=pitd_texp_eta_n_dA
            dB_NLO=pitd_texp_eta_n_dB
            dA_Tree=pitd_texp_eta_n_treelevel_dA
            dB_Tree=pitd_texp_eta_n_treelevel_dB
        return dA_NLO, dB_NLO, dA_Tree, dB_Tree



#####################################################################################
# Class to Manage Correlations & X-Correlations between data & some fit param set
#####################################################################################
class dataParamAggregate():
    def __init__(self,fitParamsAll,h5Data,zcut,pcut,comp,cfgs):
        self.fitParamsAll = fitParamsAll
        self.data = h5Data
        self.zcut = zcut
        self.pcut = pcut
        self.comp = comp # Re or Im
        self.cfgs = cfgs
        
        self.correlated = True
        self.covView=None

        self.zmin = int(zcut.split('.')[0])
        self.zmax = int(zcut.split('.')[1])
        self.Z = [zi for zi in range(self.zmin,self.zmax+1)]
        self.pmin = int(pcut.split('.')[0])
        self.pmax = int(pcut.split('.')[1])

        self.datKeys=[]
        for zi in range(self.zmin, self.zmax+1):
            for pi in range(self.pmin, self.pmax+1):
                self.datKeys.append("z_{%i} p_{%i}"%(zi,pi))

        self.dataDict = {k: [] for k in self.datKeys}

        self.avg=self.parse()
        self.cov=self.datParamCov()
        

    # Parse data & fit params
    def parse(self):
        h5In = h5py.File(self.data,'r')
        dumDict={} # dict of avg data/params to be filled and returned
        
        for z in self.Z:
            ztag="zsep%d"%z
            for m in range(self.pmin,self.pmax+1):
                ptag="pz%d"%m

                key="z_{%i} p_{%i}"%(z,m)
                
                ioffeTime = -1
                avgMat = 0.0
                avgMatErr = 0.0
                for g in range(0,self.cfgs):
                    ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                          (insertions[dirac]['Redstar'],ztag,ptag,self.comp,'pitd')][g]
                    avgMat += mat
                    dumDict[key].append(mat)
                avgMat *= (1.0/self.cfgs)
               
                # Pack this average and error into the paramDatDict
                dumDict.update({key: avgMat})
        # Now we have the average data
        
        # Form the ensemble avg of each fit parameter, by dividing by number of gauge configs
        for k, v in self.fitParamsAll.items():
            # Sum all params, skipping 'norm' if it is None
            dum=0.0
            if v is None:
                continue
            else:
                for n in v:
                    dum+=n
            dum/=self.cfgs
            dumDict.update({k: dum})

        return dumDict


    def datParamCov(self):
        # Form data/param covariance matrix + its easy matview (covView)
        self.covView=np.zeros((len(self.avg),len(self.avg)))
        for ni, ki in enumerate(self.avg.keys()):
            for nj, kj in enumerate(self.avg.keys()):
                key=(ki,kj)
                dum=0.0
                for g in range(0,self.cfgs):

                    iVal = None; jVal = None;
                    if self.fitParamsAll.has_key(ki):
                        iVal = self.fitParamsAll[ki][g]
                    else:
                        iVal = self.dataDict[ki][g]
                    if self.fitParamsAll.has_key(kj):
                        jVal = self.fitParamsAll[kj][g]
                    else:
                        jVal = self.dataDict[kj][g]
                    
                    dum+=(iVal-self.avg[ki])*(jVal-self.avg[kj])
                dum*=((1.0*(self.cfgs-1))/self.cfgs)

                if not(self.correlated) and ki != kj:
                    self.cov.update({key: 0.0})
                    self.covView[ni,nj]=0.0
                else:
                    self.cov.update({key: dum})
                    self.covView[ni,nj]=dum

        # Lastly normalize pCovView to diagonal entries
        diag=[self.covView[d,d] for d in range(0,len(self.covView))]
        for i in range(0,len(self.covView)):
            for j in range(0,len(self.covView)):
                self.covView[i,j]/=np.sqrt(diag[i]*diag[j])
    

class fits:
    def __init__(self,low,high):
        self.avg=np.linspace(low,high,500)
        self.err=np.linspace(low,high,500)


class utils:
    # Parse the fit parameters
    def parse(self):
        # Form the ensemble avg of each fit parameter, by dividing by number of gauge configs
        for k, v in self.fitParamsAll.items():
            # Sum all params, skipping 'norm' if it is None
            dum=0.0
            if v is None:
                continue
            else:
                for n in v:
                    dum+=n
            dum/=self.cfgs
            self.pAvg.update({k: dum})

        # Set normalization to beta functions, if 'norm' is not passed as a number
        # Beta fns normalization set by member numParams
        if self.fitParamsAll['norm'] is None:
            if self.numParams == 2:
                self.norm=(1.0/spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1))
            if self.numParams == 3:
                self.norm=(1.0/(spec.beta(self.pAvg['alpha']+1,\
                                      self.pAvg['beta']+1)+self.pAvg['delta']*\
                            spec.beta(self.pAvg['alpha']+2,self.pAvg['beta']+1)))
            if self.numParams == 4:
                self.norm=(1.0/(spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)\
                                +self.pAvg['gamma']*spec.beta(self.pAvg['alpha']+1.5,\
                                                              self.pAvg['beta']+1)\
                                +self.pAvg['delta']*spec.beta(self.pAvg['alpha']+2,\
                                                              self.pAvg['beta']+1)))
        else:
            self.norm=self.pAvg['norm']

        # Form parameter covariance matrix
        for ki in self.pAvg.keys():
            for kj in self.pAvg.keys():
                key=(ki,kj)
                dum=0.0
                for g in range(0,self.cfgs):
                    dum+=(self.fitParamsAll[ki][g]-self.pAvg[ki])*\
                          (self.fitParamsAll[kj][g]-self.pAvg[kj])
                dum*=((1.0*(self.cfgs-1))/self.cfgs)

                if not(self.correlated) and ki != kj:
                    self.pCov.update({key: 0.0})
                else:
                    self.pCov.update({key: dum})
                    

                
    # Print out fit parameter covariance
    def printFitCov(self):
        for k, v in self.pCov.items():
            print("%s = %.7f"%(k,v)) #print "%s = %.7f"%(k,v)


    # Print out fit parameters determined from jackknife
    def printFitParams(self):
        # print "Average fit parameters for %s"%self.name
        for k in self.pAvg.keys():
            # print "     %s (%s) = %.7f +/- %.7f"%(k,self.name,self.pAvg[k],\
            #                                       np.sqrt(self.pCov[(k,k)]))
            print("     %s (%s) = %.7f +/- %.7f"%(k,self.name,self.pAvg[k],\
                                                  np.sqrt(self.pCov[(k,k)])))

    # Print normalization as a check
    def printNorm(self):
        # print "%d-param q%s Normalization = %.7f"%(self.numParams,self.name,self.norm)
        print("%d-param q%s Normalization = %.7f"%(self.numParams,self.name,self.norm))


    # Integrate the PDF for a check
    def quarkSum(self):
        s = lambda x: self.pdf(x)
        # print "Area under %d-param q%s PDF = %.7f "%(self.numParams,self.name,integrate.quad(s,0,1)[0])
        print("Area under %d-param q%s PDF = %.7f "%\
              (self.numParams,self.name,integrate.quad(s,0,1)[0]))

    # Evaluate the ITD
    def ITD(self,nu):
        res_r=[]
        res_i=[]
        for nu_i in nu:
            r = lambda x: np.cos(nu_i*x)*self.pdf(x)
            i = lambda x: np.sin(nu_i*x)*self.pdf(x)
            res_r.append(integrate.quad(r,0,1)[0])
            res_i.append(integrate.quad(i,0,1)[0])
        return res_r, res_i
        
    # Plot pdf fits
    def plotPDF(self,ax):
        if self.name == 'bar':
            lab=r'$%s_{\overline{q}/N}(x,\mu^2)^{{\rm n}=%d}_{%s}$'%(self.pdfVariety,self.numParams,self.kernel)
        else:
            lab=r'$%s_{q_{\rm %s}/N}(x,\mu^2)^{{\rm n}=%d}_{%s}$'%(self.pdfVariety,self.name,self.numParams,self.kernel)
            
        ax.plot(self.x,self.pdf(self.x),color=self.col,label=lab)
        ax.fill_between(self.x,self.pdf(self.x)+self.pdfError(self.x),\
                        self.pdf(self.x)-self.pdfError(self.x),color=self.col,\
                        alpha=0.3,label=lab,lw=0)

    # Plot xpdf fits
    def plotXPDF(self,ax):
        if self.name == 'bar':
            lab=r'$x%s_{\overline{q}/N}(x,\mu^2)^{{\rm n}=%d}_{%s}$'%(self.pdfVariety,self.numParams,self.kernel)
        else:
            lab=r'$x%s_{q_{\rm %s}/N}(x,\mu^2)^{{\rm n}=%d}_{%s}$'%(self.pdfVariety,self.name,self.numParams,\
                                                                   self.kernel)
            
        ax.plot(self.x,self.x*self.pdf(self.x),color=self.col,label=lab)
        ax.fill_between(self.x,self.x*self.pdf(self.x)+self.x*self.pdfError(self.x),\
                        self.x*self.pdf(self.x)-self.x*self.pdfError(self.x),\
                        color=self.col,alpha=0.3,label=lab,lw=0)

    # Plot ITD
    def plotITDReal(self,ax):
        ax.plot(self.nu,self.ITD(self.nu)[0],\
                label=r'$\mathfrak{Re}\ Q(\nu,\mu^2)^{{\rm n}=%d}_{%s}$'%(self.numParams,self.kernel),\
                color=self.col,alpha=0.3) 
        ax.fill_between(self.nu,[x+y for x, y in\
                                 zip(self.ITD(self.nu)[0],self.ITDErrorReal(self.nu))],\
                        [x-y for x, y in zip(self.ITD(self.nu)[0],self.ITDErrorReal(self.nu))],\
                        label=r'$\mathfrak{Re}\ Q(\nu,\mu^2)^{{\rm n}=%d}_{%s}$'%\
                        (self.numParams,self.kernel),color=self.col,alpha=0.3,lw=0)


    # Plot Imag ITD
    def plotITDImag(self,ax):
        ax.plot(self.nu,self.ITD(self.nu)[1],\
                label=r'$\mathfrak{Im}\ Q(\nu,\mu^2)^{{\rm n}=%d}_{%s}$'%\
                (self.numParams,self.kernel),color=self.col,alpha=0.3)
        ax.fill_between(self.nu,[x+y for x, y in\
                                 zip(self.ITD(self.nu)[1],self.ITDErrorImag(self.nu))],\
                        [x-y for x, y in zip(self.ITD(self.nu)[1],
                                             self.ITDErrorImag(self.nu))],\
                        label=r'$\mathfrak{Im}\ Q(\nu,\mu^2)^{{\rm n}=%d}_{%s}$'%\
                        (self.numParams,self.kernel),color=self.col,alpha=0.3,lw=0)


    # Write PDF to File
    def writePDF(self):
        zippy = zip(self.x,self.pdf(self.x),self.pdfError(self.x))
        np.savetxt('%sparam-PDF.q%s.fit.dat'%(self.numParams,self.name), list(zippy),\
                   fmt='%.7f %.7f %.7f')

    # Write ITD to File
    def writeITD_RE(self):
        zippy = zip(self.nu, self.ITD(self.nu)[0], self.ITDErrorReal(self.nu))
        # nb. need to force zip to evaluate w/ list(...) in python3
        np.savetxt('%sparam-ITD.q%s.fit.dat'%(self.numParams,self.name), list(zippy),\
                   fmt='%.7f %.7f %.7f')
    def writeITD_IM(self):
        zippy = zip(self.nu, self.ITD(self.nu)[0], self.ITDErrorImag(self.nu))
        # nb. need to force zip to evaluate w/ list(...) in python3
        np.savetxt('%sparam-ITD.q%s.fit.dat'%(self.numParams,self.name), list(zippy),\
                   fmt='%.7f %.7f %.7f')
        

    # Add pdf fit parameters to pdf plots
    def addParamsToPlot(self,ax,buff):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xspot=xmax-0.4*(xmax-xmin)
        yspot=ymax-0.1*buff*(ymax-ymin)
        ydiff=0.025*(ymax-ymin)

        
        cnt=-1
        for KEY, VAL in self.pAvg.items():
            cnt+=1
            if KEY == 'norm':
                ax.text(xspot,yspot-cnt*ydiff,r'$N^{%s}_{%d} = %.3f(%.3f)_{%s}$'%
                        (self.name,self.numParams,VAL,np.sqrt(self.pCov[(KEY,KEY)]),self.kernel),\
                        fontsize=13)
            elif KEY == 'chi2':
                ax.text(xspot,yspot-cnt*ydiff,r'$[\chi_r^2]^{%s}_{%d} = %.3f(%.3f)_{%s}$'%
                        (self.name,self.numParams,VAL,np.sqrt(self.pCov[(KEY,KEY)]),self.kernel),\
                        fontsize=13)
            else:
                ax.text(xspot,yspot-cnt*ydiff,r'$\%s^{%s}_{%d} = %.3f(%.3f)_{%s}$'%
                        (KEY,self.name,self.numParams,VAL,np.sqrt(self.pCov[(KEY,KEY)]),self.kernel),\
                        fontsize=13)

        # buff=yspot-buff-cnt*ydiff
        # print "BUFF = %.4f"%buff
        # return buff

        
###################################################################
###################################################################
########### CLASS TO MANAGE 2-PARAMETER FITS TO ITDs ##############
class pdf2(utils):
    # Variables shared by all such instances
    x=np.linspace(xMin,xMax,500)
    nu=np.linspace(0,20,500)
    pdffit=fits(xMin,xMax)
    itdfit=fits(0,20)
    numParams=2


    def __init__(self,name,kernel,color,fitParamsAll,correlated,cfgs):
        self.name=name
        self.kernel=kernel
        self.fitParamsAll=fitParamsAll
        self.pAvg={}
        self.pCov={}
        self.col=color
        # self.cfgs=np.size(fitParamsAll.items()[0][1])
        self.cfgs=cfgs
        self.norm=None
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.correlated=correlated

        
    # The 2-parameter PDF
    def pdf(self,x):
        return self.norm*np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])


    # Determine the error of pdf fit
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})
        print(partials) # print partials


        # PDF partials for the valence case
        if self.fitParamsAll['norm'] is None:
            partials['alpha']=(self.pdf(x)*(-H(self.pAvg['alpha'])+\
                                            H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x)))
            partials['beta']=(self.pdf(x)*(-H(self.pAvg['beta'])+\
                                           H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(1-x)))
        # PDF partials for the q+ distribution
        else:
            partials['alpha']=self.pdf(x)*np.log(x)
            partials['beta']=self.pdf(x)*np.log(1-x)
            partials['norm']=np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])

        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=partials[i]*self.pCov[(i,j)]*partials[j]
        return np.sqrt(error)
    


    # Determine the error of Real ITD fit
    def ITDErrorReal(self,nu):
        real_partials={}
        for k in self.pAvg.keys():
            real_partials.update({k: 0.0})
            
        res_r=[]
        for nu_i in nu:
            qar = lambda x : np.cos(nu_i*x)*(self.pdf(x)*(-H(self.pAvg['alpha'])+\
                                        H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x)))
            qbr = lambda x : np.cos(nu_i*x)*(self.pdf(x)*(-H(self.pAvg['beta'])+\
                                       H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(1-x)))

            real_partials['alpha']=integrate.quad(qar,0,1)[0]
            real_partials['beta']=integrate.quad(qbr,0,1)[0]

            error_r=0.0
            for i in self.pAvg:
                for j in self.pAvg:
                    error_r+=real_partials[i]*self.pCov[(i,j)]*real_partials[j]

            res_r.append(np.sqrt(error_r))

        return res_r


    # Determine the error of Imag ITD fit
    def ITDErrorImag(self,nu):
        imag_partials={}
        for k in self.pAvg.keys():
            imag_partials.update({k: 0.0})

        res_i=[]
        for nu_i in nu:
            qai = lambda x : np.sin(nu_i*x)*(self.pdf(x)*np.log(x))

            qbi = lambda x : np.sin(nu_i*x)*(self.pdf(x)*np.log(1-x))
            
            qni = lambda x : np.sin(nu_i*x)*(np.power(x,self.pAvg['alpha'])\
                                             *np.power(1-x,self.pAvg['beta']))

            

            imag_partials['alpha']=integrate.quad(qai,0,1)[0]
            imag_partials['beta']=integrate.quad(qbi,0,1)[0]
            imag_partials['norm']=integrate.quad(qni,0,1)[0]

            error_i=0.0
            for i in self.pAvg:
                for j in self.pAvg:
                    error_i+=imag_partials[i]*self.pCov[(i,j)]*imag_partials[j]

            res_i.append(np.sqrt(error_i))

        return res_i



#########
# Results of phenomenological/theory fits from literature
#########

    
###################################################################
###################################################################
################ CLASS TO MANAGE CJ15 Q-VALENCE ###################
class CJ15(utils):
    # Variables shared by all instances
    x=np.linspace(xMin,xMax,500)
    pdffit=fits(xMin,xMax)
    numParams=4

    def __init__(self,name,kernel,color,pAvg,pErr,pAvg2=None,pErr2=None):
        self.name=name
        self.kernel=kernel
        self.col=color
        self.pAvg=pAvg
        self.pErr=pErr
        self.pAvg2=pAvg2
        self.pErr2=pErr2

    # The CJ15 PDF Form
    def pdf(self,x):
        if self.pAvg2 is not None and self.pErr2 is not None:
            return self.pAvg['a0']*np.power(x,self.pAvg['a1']-1)*np.power(1-x,self.pAvg['a2'])\
                *(1+self.pAvg['a3']*np.sqrt(x)+self.pAvg['a4']*x)\
                -self.pAvg2['a0']*np.power(x,self.pAvg2['a1']-1)*np.power(1-x,self.pAvg2['a2'])\
                *(1+self.pAvg2['a3']*np.sqrt(x)+self.pAvg2['a4']*x)
        else:
            return self.pAvg['a0']*np.power(x,self.pAvg['a1']-1)*np.power(1-x,self.pAvg['a2'])\
                *(1+self.pAvg['a3']*np.sqrt(x)+self.pAvg['a4']*x)
    
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})

        partials['a0']=np.power(x,self.pAvg['a1']-1)*np.power(1-x,self.pAvg['a2'])\
                *(1+self.pAvg['a3']*np.sqrt(x)+self.pAvg['a4']*x)
        partials['a1']=self.pAvg['a0']*np.power(x,self.pAvg['a1']-1)*np.power(1-x,self.pAvg['a2'])\
                *(1+self.pAvg['a3']*np.sqrt(x)+self.pAvg['a4']*x)*np.log(x)
        partials['a2']=self.pAvg['a0']*np.power(x,self.pAvg['a1']-1)*np.power(1-x,self.pAvg['a2'])\
                *(1+self.pAvg['a3']*np.sqrt(x)+self.pAvg['a4']*x)*np.log(1-x)
        partials['a3']=self.pAvg['a0']*np.power(x,self.pAvg['a1']-0.5)*np.power(1-x,self.pAvg['a2'])
        partials['a4']=self.pAvg['a0']*np.power(x,self.pAvg['a1'])*np.power(1-x,self.pAvg['a2'])
    
        error=0.0
        for k,v in self.pErr.items():
            error+=np.power(partials[k],2)*np.power(v,2)

        # If a second set of params is passed, modify the partials and include them in error
        if self.pAvg2 is not None and self.pErr2 is not None:
            for k in self.pAvg2.keys():
                partials.update({k: 0.0})

            partials['a0']=np.power(x,self.pAvg2['a1']-1)*np.power(1-x,self.pAvg2['a2'])\
                            *(1+self.pAvg2['a3']*np.sqrt(x)+self.pAvg2['a4']*x)
            partials['a1']=self.pAvg2['a0']*np.power(x,self.pAvg2['a1']-1)\
                            *np.power(1-x,self.pAvg2['a2'])\
                            *(1+self.pAvg2['a3']*np.sqrt(x)+self.pAvg2['a4']*x)*np.log(x)
            partials['a2']=self.pAvg2['a0']*np.power(x,self.pAvg2['a1']-1)\
                            *np.power(1-x,self.pAvg2['a2'])\
                            *(1+self.pAvg2['a3']*np.sqrt(x)+self.pAvg2['a4']*x)*np.log(1-x)
            partials['a3']=self.pAvg2['a0']*np.power(x,self.pAvg2['a1']-0.5)\
                            *np.power(1-x,self.pAvg2['a2'])
            partials['a4']=self.pAvg2['a0']*np.power(x,self.pAvg2['a1'])\
                            *np.power(1-x,self.pAvg2['a2'])
            
            for k,v in self.pErr2.items():
                error+=np.power(partials[k],2)*np.power(v,2) 
                
                
        return np.sqrt(error)


    
    

###################################################################
###################################################################
########### CLASS TO MANAGE 2-PARAMETER DERIVED Q #################    
class Q2(utils):
    # Variables shared by all such instances
    x=np.linspace(xMin,xMax,500)
    nu=np.linspace(0,20,500)
    pdffit=fits(xMin,xMax)
    itdfit=fits(0,20)
    numParams=2

    def __init__(self,name,kernel,color,fitParamsAll,correlated,cfgs):
        self.name=name
        self.kernel=kernel
        self.fitParamsAll=fitParamsAll
        self.pAvg={}
        self.pCov={}
        self.col=color
        self.cfgs=cfgs
        self.norm=None
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.correlated=correlated

    # The 2-parameter q-pdf
    def pdf(self,x):
        return 0.5*( (np.power(x,self.pAvg['alpha'])\
                      *np.power(1-x,self.pAvg['beta']))/(spec.beta(self.pAvg['alpha']+1,\
                                                                   self.pAvg['beta']+1))\
                     +self.pAvg['norm']*np.power(x,self.pAvg['alphaP'])\
                     *np.power(1-x,self.pAvg['betaP']) )

    # Determine the 2-parameter pdf error
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})

        # PDF partials
        partials['alpha']=(((np.power(x,self.pAvg['alpha'])\
                             *np.power(1-x,self.pAvg['beta']))\
                            /(2*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)))\
                           *(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                             +np.log(x)))
        
        partials['beta']=(((np.power(x,self.pAvg['alpha'])\
                            *np.power(1-x,self.pAvg['beta']))\
                           /(2*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)))\
                          *(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                            +np.log(1-x)))

        partials['normP']=0.5*np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])
        
        partials['alphaP']=0.5*self.pAvg['norm']*np.power(x,self.pAvg['alpha'])\
                            *np.power(1-x,self.pAvg['beta'])*np.log(x)
        
        partials['betaP']=0.5*self.pAvg['norm']*np.power(x,self.pAvg['alpha'])\
                           *np.power(1-x,self.pAvg['beta'])*np.log(1-x)
            
        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=partials[i]*self.pCov[(i,j)]*partials[j]
        return np.sqrt(error)



###################################################################
###################################################################
########### CLASS TO MANAGE 2-PARAMETER DERIVED QBAR ##############    
class QBar2(utils):
    # Variables shared by all such instances
    x=np.linspace(xMin,xMax,500)
    nu=np.linspace(0,20,500)
    pdffit=fits(xMin,xMax)
    itdfit=fits(0,20)
    numParams=2

    def __init__(self,name,kernel,color,fitParamsAll,correlated,cfgs):
        self.name=name
        self.kernel=kernel
        self.fitParamsAll=fitParamsAll
        self.pAvg={}
        self.pCov={}
        self.col=color
        self.cfgs=cfgs
        self.norm=None
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.correlated=correlated

    # The 2-parameter q-pdf
    def pdf(self,x):
        return 0.5*( -(np.power(x,self.pAvg['alpha'])\
                       *np.power(1-x,self.pAvg['beta']))/(spec.beta(self.pAvg['alpha']+1,\
                                                                    self.pAvg['beta']+1))\
                     +self.pAvg['norm']*np.power(x,self.pAvg['alphaP'])\
                     *np.power(1-x,self.pAvg['betaP']) )

    # Determine the 2-parameter pdf error
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})

        # PDF partials
        partials['alpha']=(((np.power(x,self.pAvg['alpha'])\
                             *np.power(1-x,self.pAvg['beta']))\
                            /(2*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)))\
                           *(H(self.pAvg['alpha'])-H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                             -np.log(x)))
        
        partials['beta']=(((np.power(x,self.pAvg['alpha'])\
                            *np.power(1-x,self.pAvg['beta']))\
                           /(2*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)))\
                          *(H(self.pAvg['beta'])-H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                            -np.log(1-x)))

        partials['normP']=0.5*np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])
        
        partials['alphaP']=0.5*self.pAvg['norm']*np.power(x,self.pAvg['alpha'])\
                            *np.power(1-x,self.pAvg['beta'])*np.log(x)
        
        partials['betaP']=0.5*self.pAvg['norm']*np.power(x,self.pAvg['alpha'])\
                           *np.power(1-x,self.pAvg['beta'])*np.log(1-x)
            
        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=partials[i]*self.pCov[(i,j)]*partials[j]
        return np.sqrt(error)
    

    
###################################################################
###################################################################
########### CLASS TO MANAGE 3-PARAMETER FITS TO ITDs ##############
class pdf3(utils):
    # Variables shared by all such instances
    x=np.linspace(xMin,xMax,500)
    nu=np.linspace(0,20,500)
    pdffit=fits(xMin,xMax)
    itdfit=fits(0,20)
    numParams=3

    def __init__(self,name,kernel,color,fitParamsAll,correlated,cfgs):
        self.name=name
        self.kernel=kernel
        self.fitParamsAll=fitParamsAll
        self.pAvg={}
        self.pCov={}
        self.col=color
        self.cfgs=cfgs
        self.norm=None
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.correlated=correlated

        
    # The 3-parameter PDF
    def pdf(self,x):
        return self.norm*np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])\
            *(1+self.pAvg['delta']*x)

    # Determine the error of pdf fit
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})

        # PDF partials for the valence case
        if self.fitParamsAll['norm'] is None:
            partials['alpha']=(self.pdf(x)*self.norm)*(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])*(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x))+self.pAvg['delta']*spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta'])*(-H(1+self.pAvg['alpha'])+H(2+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x)))

            partials['beta']=(self.pdf(x)*self.norm)\
                                 *(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   *(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                                     +np.log(1-x))+self.pAvg['delta']\
                                   *spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   *(-H(self.pAvg['beta'])+H(2+self.pAvg['alpha']\
                                                             +self.pAvg['beta'])+np.log(1-x)))
                                 
            partials['delta']=((self.pdf(x)*self.norm)/(1+self.pAvg['delta']*x))\
                                 *(x*spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   -spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta']))
                                 
        # PDF partials for the q+ distribution
        else:
            partials['alpha']=self.pdf(x)*np.log(x)
            partials['beta']=self.pdf(x)*np.log(1-x)
            partials['delta']=self.norm*np.power(x,self.pAvg['alpha']+1)\
                                 *np.power(1-x,self.pAvg['beta'])
            partials['norm']=np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])\
                                 *(1+self.pAvg['delta']*x)
            

        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                if not(self.correlated):
                    if i == j:
                        error+=partials[i]*self.pCov[(i,j)]*partials[j]
                    else:
                        error+=0.0
                else:
                    error+=partials[i]*self.pCov[(i,j)]*partials[j]
        return np.sqrt(error)
    


    # Determine the error of Real ITD fit
    def ITDErrorReal(self,nu):
        real_partials={}
        for k in self.pAvg.keys():
            real_partials.update({k: 0.0})
            
        res_r=[]
        for nu_i in nu:
            qar = lambda x : np.cos(nu_i*x)*(self.pdf(x)*self.norm)*(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])*(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x))+self.pAvg['delta']*spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta'])*(-H(1+self.pAvg['alpha'])+H(2+self.pAvg['alpha']+self.pAvg['beta'])+np.log(x)))
            
            qbr = lambda x : np.cos(nu_i*x)*(self.pdf(x)*self.norm)\
                                 *(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   *(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                                     +np.log(1-x))+self.pAvg['delta']\
                                   *spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   *(-H(self.pAvg['beta'])+H(2+self.pAvg['alpha']\
                                                             +self.pAvg['beta'])+np.log(1-x)))
            
            qdr = lambda x : np.cos(nu_i*x)*((self.pdf(x)*self.norm)/(1+self.pAvg['delta']*x))\
                                 *(x*spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                   -spec.beta(2+self.pAvg['alpha'],1+self.pAvg['beta']))


            real_partials['alpha']=integrate.quad(qar,0,1)[0]
            real_partials['beta']=integrate.quad(qbr,0,1)[0]
            real_partials['delta']=integrate.quad(qdr,0,1)[0]

            
            error_r=0.0
            for i in self.pAvg:
                for j in self.pAvg:
                    if not(self.correlated):
                        if i == j:
                            error_r+=real_partials[i]*self.pCov[(i,j)]*real_partials[j]
                        else:
                            error_r+=0.0
                    else:
                        error_r+=real_partials[i]*self.pCov[(i,j)]*real_partials[j]
            res_r.append(np.sqrt(error_r))

        return res_r


    # Determine the error of Imag ITD fit
    def ITDErrorImag(self,nu):
        imag_partials={}
        for k in self.pAvg.keys():
            imag_partials.update({k: 0.0})

        res_i=[]
        for nu_i in nu:
            qai = lambda x : np.sin(nu_i*x)*self.pdf(x)*np.log(x)
            qbi = lambda x : np.sin(nu_i*x)*self.pdf(x)*np.log(1-x)
            qdi = lambda x : np.sin(nu_i*x)*self.norm*np.power(x,self.pAvg['alpha']+1)\
                                 *np.power(1-x,self.pAvg['beta'])
            qni = lambda x : np.sin(nu_i*x)*np.power(x,self.pAvg['alpha'])\
                  *np.power(1-x,self.pAvg['beta'])*(1+self.pAvg['delta']*x)



            imag_partials['alpha']=integrate.quad(qai,0,1)[0]
            imag_partials['beta']=integrate.quad(qbi,0,1)[0]
            imag_partials['delta']=integrate.quad(qdi,0,1)[0]
            imag_partials['norm']=integrate.quad(qni,0,1)[0]


            error_i=0.0
            for i in self.pAvg:
                for j in self.pAvg:
                    if not(self.correlated):
                        if i == j:
                            error_i+=imag_partials[i]*self.pCov[(i,j)]*imag_partials[j]
                        else:
                            error_i+=0.0
                    else:
                        error_i+=imag_partials[i]*self.pCov[(i,j)]*imag_partials[j]
            res_i.append(np.sqrt(error_i))

        return res_i



###################################################################
###################################################################
########### CLASS TO MANAGE 4-PARAMETER FITS TO ITDs ##############
class pdf4(utils):
    # Variables shared by all such instances
    x=np.linspace(xMin,xMax,500)
    nu=np.linspace(0,20,500)
    pdffit=fits(xMin,xMax)
    itdfit=fits(0,20)
    numParams=4

    def __init__(self,name,kernel,color,fitParamsAll,correlated):
        self.name=name
        self.kernel=kernel
        self.fitParamsAll=fitParamsAll
        self.pAvg={}
        self.pCov={}
        self.col=color
        self.cfgs=np.size(fitParamsAll.items()[0][1])
        self.norm=None
        self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
        self.correlated=correlated

        
    # The 4-parameter PDF
    def pdf(self,x):
        return self.norm*np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])\
            *(1+self.pAvg['gamma']*np.sqrt(x)+self.pAvg['delta']*x)

    # Determine the error of pdf fit
    def pdfError(self,x):
        partials={}
        # Make a small dictionary to access partials of pdf
        for k in self.pAvg.keys():
            partials.update({k: 0.0})

        # PDF partials for the valence case
        if self.fitParamsAll['norm'] is None:
            partials['alpha']=(self.pdf(x)*self.norm)\
                              *(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                *(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                                  +np.log(x))+self.pAvg['delta']*spec.beta(2+self.pAvg['alpha'],\
                                                                             1+self.pAvg['beta'])\
                                *(-H(1+self.pAvg['alpha'])+H(2+self.pAvg['alpha']+self.pAvg['beta'])\
                                  +np.log(x))\
                                +self.pAvg['gamma']*spec.beta(1.5+self.pAvg['alpha'],\
                                                              1+self.pAvg['beta'])\
                                *(np.log(x)-spec.polygamma(0,1.5+self.pAvg['alpha'])\
                                  +spec.polygamma(0,2.5+self.pAvg['alpha']+self.pAvg['beta'])))
            
            partials['beta']=(self.pdf(x)*self.norm)\
                              *(spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta'])\
                                *(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
                                  +np.log(1-x))+self.pAvg['delta']*spec.beta(2+self.pAvg['alpha'],\
                                                                             1+self.pAvg['beta'])\
                                *(-H(self.pAvg['beta'])+H(2+self.pAvg['alpha']+self.pAvg['beta'])\
                                  +np.log(1-x))\
                                +self.pAvg['gamma']*spec.beta(1.5+self.pAvg['alpha'],\
                                                              1+self.pAvg['beta'])\
                                *(np.euler_gamma-H(self.pAvg['beta'])+np.log(1-x)\
                                  +spec.polygamma(0,2.5+self.pAvg['alpha']+self.pAvg['beta'])))
                                
            partials['gamma']=((self.norm*self.pdf(x))\
                               /(1+self.pAvg['gamma']*np.sqrt(x)+self.pAvg['delta']*x))\
                               *(np.sqrt(x)*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)\
                                 -(1+self.pAvg['delta']*x)*spec.beta(self.pAvg['alpha']+1.5,\
                                                                 self.pAvg['beta']+1)\
                                 +self.pAvg['delta']*np.sqrt(x)*spec.beta(self.pAvg['alpha']+2,\
                                                                          self.pAvg['beta']+1))
            partials['delta']=((self.norm*self.pdf(x))\
                               /(1+self.pAvg['gamma']*np.sqrt(x)+self.pAvg['delta']*x))\
                               *(x*spec.beta(self.pAvg['alpha']+1,self.pAvg['beta']+1)\
                                 +self.pAvg['gamma']*x*spec.beta(self.pAvg['alpha']+1.5,\
                                                                 self.pAvg['beta']+1)\
                                 -(1+self.pAvg['gamma']*np.sqrt(x))*spec.beta(self.pAvg['alpha']+2,\
                                                                              self.pAvg['beta']+1))
                                                       
                                 
        # PDF partials for the q+ distribution
        else:
            partials['alpha']=self.pdf(x)*np.log(x)
            partials['beta']=self.pdf(x)*np.log(1-x)
            partials['gamma']=self.norm*np.power(x,self.pAvg['alpha']+0.5)\
                               *np.power(1-x,self.pAvg['beta'])
            partials['delta']=self.norm*np.power(x,self.pAvg['alpha']+1)\
                               *np.power(1-x,self.pAvg['beta'])
            partials['norm']=np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])\
                                 *(1+self.pAvg['gamma']*np.sqrt(x)+self.pAvg['delta']*x)

        error=0.0
        for i in self.pAvg:
            for j in self.pAvg:
                error+=partials[i]*self.pCov[(i,j)]*partials[j]
        return np.sqrt(error)


    

# ###################################################################
# ###################################################################
# ########### CLASS TO MANAGE 4-PARAMETER JACOBI FITS TO ITDs ##############
# class jacobi4(utils):
#     # Variables shared by all such instances
#     x=np.linspace(xMin,xMax,500)
#     nu=np.linspace(0,20,500)
#     pdffit=fits(xMin,xMax)
#     itdfit=fits(0,20)
#     numParams=4

#     def __init__(self,name,kernel,color,fitParamsAll,correlated):
#         self.name=name
#         self.kernel=kernel
#         self.fitParamsAll=fitParamsAll
#         self.pAvg={}
#         self.pCov={}
#         self.col=color
#         self.cfgs=np.size(fitParamsAll.items()[0][1])
#         self.norm=None
#         self.vshift=0.0 # To shift text boxes of fit params displayed w/ pdfs
#         self.correlated=correlated

        
#     # The 4-parameter Jacobi PDF
#     def pdf(self,x):
#         return np.power(x,self.pAvg['alpha'])*np.power(1-x,self.pAvg['beta'])\
#             *( (1.0/spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta']))\
#                +self.pAvg['gamma']*jacobi(1,self.pAvg['alpha'],self.pAvg['beta'],x)\
#                +self.pAvg['delta']*jacobi(2,self.pAvg['alpha'],self.pAvg['beta'],x))


#     # Determine the error of pdf fit
#     def pdfError(self,x):
#         partials={}
#         # Make a small dictionary to access partials of pdf
#         for k in self.pAvg.keys():
#             partials.update({k: 0.0})

#         # PDF partials for the valence case
#         if self.fitParamsAll['norm'] is None:
#             partials['alpha']=0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *((-1+x)*(-2*self.pAvg['gamma']+self.pAvg['delta']\
#                                          *(-3-2*self.pAvg['alpha']+7*x\
#                                            +2*(self.pAvg['alpha']+self.pAvg['beta'])*x))\
#                                  +((1+self.pAvg['alpha'])*((2+self.pAvg['alpha'])*self.pAvg['delta']\
#                                                            +2*self.pAvg['gamma'])\
#                                    -2*((2+self.pAvg['alpha'])*(3+self.pAvg['alpha']+self.pAvg['beta'])\
#                                        *self.pAvg['delta']+(2+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['gamma'])\
#                                    *x+(3+self.pAvg['alpha']+self.pAvg['beta'])*(4+self.pAvg['alpha']+self.pAvg['beta'])\
#                                    *self.pAvg['delta']*np.power(x,2))*np.log(x)\
#                                  +(2*(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
#                                       +np.log(x)))/spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta']))
            
#             partials['beta']=0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                               *(x*(-2*((2+self.pAvg['alpha'])*self.pAvg['delta']+self.pAvg['gamma'])\
#                                    +(7+2*self.pAvg['alpha']+2*self.pAvg['beta'])*self.pAvg['delta']*x)\
#                                 +((1+self.pAvg['alpha'])*((2+self.pAvg['alpha'])*self.pAvg['delta']+2*self.pAvg['gamma'])\
#                                   -2*((2+self.pAvg['alpha'])*(3+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['delta']\
#                                       +(2+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['gamma'])*x+(3+self.pAvg['alpha']\
#                                                                                                        +self.pAvg['beta'])\
#                                   *(4+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['delta']*np.power(x,2))*np.log(1-x)\
#                                 +(2*(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
#                                      +np.log(1-x)))/spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta']))
                                
#             partials['gamma']=np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *(1+self.pAvg['alpha']-(2+self.pAvg['alpha']+self.pAvg['beta'])*x)
            
#             partials['delta']=0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *((1+self.pAvg['alpha'])*(2+self.pAvg['alpha'])-2*(2+self.pAvg['alpha'])\
#                                  *(3+self.pAvg['alpha']+self.pAvg['beta'])*x+(3+self.pAvg['alpha']\
#                                                                               +self.pAvg['beta'])\
#                                  *(4+self.pAvg['alpha']+self.pAvg['beta'])*np.power(x,2))

#         else:
#             print "Don't have derivatives of jacobi polynomials set for imag fits"
#             sys.exit()


#         error=0.0
#         for i in self.pAvg:
#             for j in self.pAvg:
#                 error+=partials[i]*self.pCov[(i,j)]*partials[j]
#         return np.sqrt(error)
        

#     # Determine the error of Real ITD fit
#     def ITDErrorReal(self,nu):
#         real_partials={}
#         for k in self.pAvg.keys():
#             real_partials.update({k: 0.0})
            
#         res_r=[]
#         for nu_i in nu:
#             qar = lambda x : np.cos(nu_i*x)*0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *((-1+x)*(-2*self.pAvg['gamma']+self.pAvg['delta']\
#                                          *(-3-2*self.pAvg['alpha']+7*x\
#                                            +2*(self.pAvg['alpha']+self.pAvg['beta'])*x))\
#                                  +((1+self.pAvg['alpha'])*((2+self.pAvg['alpha'])*self.pAvg['delta']\
#                                                            +2*self.pAvg['gamma'])\
#                                    -2*((2+self.pAvg['alpha'])*(3+self.pAvg['alpha']+self.pAvg['beta'])\
#                                        *self.pAvg['delta']+(2+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['gamma'])\
#                                    *x+(3+self.pAvg['alpha']+self.pAvg['beta'])*(4+self.pAvg['alpha']+self.pAvg['beta'])\
#                                    *self.pAvg['delta']*np.power(x,2))*np.log(x)\
#                                  +(2*(-H(self.pAvg['alpha'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
#                                       +np.log(x)))/spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta']))
            
#             qbr = lambda x : np.cos(nu_i*x)*0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                               *(x*(-2*((2+self.pAvg['alpha'])*self.pAvg['delta']+self.pAvg['gamma'])\
#                                    +(7+2*self.pAvg['alpha']+2*self.pAvg['beta'])*self.pAvg['delta']*x)\
#                                 +((1+self.pAvg['alpha'])*((2+self.pAvg['alpha'])*self.pAvg['delta']+2*self.pAvg['gamma'])\
#                                   -2*((2+self.pAvg['alpha'])*(3+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['delta']\
#                                       +(2+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['gamma'])*x+(3+self.pAvg['alpha']\
#                                                                                                        +self.pAvg['beta'])\
#                                   *(4+self.pAvg['alpha']+self.pAvg['beta'])*self.pAvg['delta']*np.power(x,2))*np.log(1-x)\
#                                 +(2*(-H(self.pAvg['beta'])+H(1+self.pAvg['alpha']+self.pAvg['beta'])\
#                                      +np.log(1-x)))/spec.beta(1+self.pAvg['alpha'],1+self.pAvg['beta']))

#             qgr = lambda x : np.cos(nu_i*x)*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *(1+self.pAvg['alpha']-(2+self.pAvg['alpha']+self.pAvg['beta'])*x)
            
#             qdr = lambda x : np.cos(nu_i*x)*0.5*np.power(1-x,self.pAvg['beta'])*np.power(x,self.pAvg['alpha'])\
#                                *((1+self.pAvg['alpha'])*(2+self.pAvg['alpha'])-2*(2+self.pAvg['alpha'])\
#                                  *(3+self.pAvg['alpha']+self.pAvg['beta'])*x+(3+self.pAvg['alpha']\
#                                                                               +self.pAvg['beta'])\
#                                  *(4+self.pAvg['alpha']+self.pAvg['beta'])*np.power(x,2))


#             real_partials['alpha']=integrate.quad(qar,0,1)[0]
#             real_partials['beta']=integrate.quad(qbr,0,1)[0]
#             real_partials['gamma']=integrate.quad(qgr,0,1)[0]
#             real_partials['delta']=integrate.quad(qdr,0,1)[0]

            
#             error_r=0.0
#             for i in self.pAvg:
#                 for j in self.pAvg:
#                     error_r+=real_partials[i]*self.pCov[(i,j)]*real_partials[j]
#             res_r.append(np.sqrt(error_r))

#         return res_r



#####################################################################
#####################################################################
########## CLASS TO MANAGE ALL HIGHER-TWIST JACOBI STUFF ############
class jacobiHT:
    def __init__(self,twist,numJac,strtIdx,pAvg,pCov,comp):
        self.T=twist
        self.numJac=numJac
        self.pitdCorrPower=int(self.T/2-1)
        self.strtIdx=strtIdx
        self.tStr='t%i'%self.T
        self.pAvg=pAvg
        self.pCov=pCov
        self.texpKernel=texpKernel(comp)

    def pdf(self,x):
        res=0.0
        # Loop over each Jacobi polynomial for this order of twist
        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            res+=self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*Jacobi(p,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
        return np.power(x,self.pAvg['\\alpha'])*np.power(1-x,self.pAvg['\\beta'])*res

    def pdfErr(self,x):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            partials.update({'C^{%s}_%s'%(self.tStr,str(p)): 0.0})
        partials['\\alpha'] += np.log(x)*self.pdf(x)
        partials['\\beta'] += np.log(1-x)*self.pdf(x)

        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            partials['\\alpha'] += np.power(x,self.pAvg['\\alpha'])*\
                np.power(1-x,self.pAvg['\\beta'])*\
                self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*\
                dAlphaJacobi(p,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['\\beta'] += np.power(x,self.pAvg['\\alpha'])*\
                np.power(1-x,self.pAvg['\\beta'])*\
                self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*\
                dBetaJacobi(p,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['C^{%s}_%s'%(self.tStr,str(p))] +=np.power(x,self.pAvg['\\alpha'])*\
                np.power(1-x,self.pAvg['\\beta'])*\
                Jacobi(p,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)

        # Propagate error and return
        # and return the total error & partials dict (used for ITD error derivation)
        return propError(self.pCov,partials), partials
            


    #------------------------------------------------------------------------------------------------------
    # Twist-self.Twist [(z^2*\Lambda^2)^self.pitdCorrPower corrections pitd for some z (Continuous in \nu)
    #------------------------------------------------------------------------------------------------------
    def pitd(self,nu,z):
        res=0.0
        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            res+=np.power(z*LambdaQCD,(2*self.pitdCorrPower))*self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*\
                self.texpKernel.Tree(p,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
        return res

    #------------------------------------------------------------------------------------------------------------
    # Twist-self.Twist [(z^2*\Lambda^2)^self.pitdCorrPower corrections pitd Error for some z (Continuous in \nu)
    #------------------------------------------------------------------------------------------------------------
    def pitd_error(self,partials,nu,z):
        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            partials.update({'C^{%s}_%d'%(self.tStr,p): 0.0})

        for p in range(self.strtIdx,self.numJac+self.strtIdx):
            partials['\\alpha'] += np.power(z*LambdaQCD,(2*self.pitdCorrPower))*self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*\
                self.texpKernel.dA_Tree(p,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
            partials['\\beta'] += np.power(z*LambdaQCD,(2*self.pitdCorrPower))*self.pAvg['C^{%s}_%s'%(self.tStr,str(p))]*\
                self.texpKernel.dB_Tree(p,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
            partials['C^{%s}_%s'%(self.tStr,str(p))] = np.power(z*LambdaQCD,(2*self.pitdCorrPower))*\
                self.texpKernel.Tree(p,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)

            
#####################################################################
#####################################################################
########### CLASS TO MANAGE JACOBI EXPANSIONS TO PITDs ##############
class pitdJacobi(texpKernel):
    # Variables shared by all such instances
    # nu=np.linspace(0,20,500)
    # nu=np.linspace(0,16,350)
    # nu=np.linspace(0,16,20)
    # nu=np.linspace(0,17,125)    
    # nu=np.linspace(0,16,350)
    
    nu=np.linspace(0,15,250)
    x=np.linspace(0,1,300)
    # nu=np.linspace(0,4,3)
    # x=np.linspace(0,1,5)

    #### Publishable
    # nu=np.linspace(0,15,377)
    # x=np.linspace(0,1,1000)

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
        self.pdfVariety='f' if self.dirac == 8 else 'g'
        self.pdfNameSuffix='' if self.dirac == 8 else '/g_A\left(\mu^2\\right)'

        self.numLT=self.pNums[0]
        self.numAZ=self.pNums[1]
        self.numHTs=[h for h in self.pNums[2:]]
        # self.numT4=self.pNums[2]
        # self.numT6=self.pNums[3]
        # self.numParams=2+self.numLT+self.numAZ+self.numT4+self.numT6
        self.numParams=2+self.numLT+self.numAZ+sum(self.numHTs)

        # Gather all higher-twist pdf/itd stuff
        self.HT=[jacobiHT(2*(n+2),h,self.cs,self.pAvg,self.pCov,self.comp) for n,h in enumerate(self.numHTs)]

        # Truncation string to include with PDF/ITD labeling
        self.truncStr="%i%i"%(self.numLT,self.numAZ)
        for n,h in enumerate(self.numHTs):
            # No longer want twist-6/twist-8 pieces, so just truncate the enumeration to keep functionality
            if n < 2:
                self.truncStr+="%i"%h
        
        # Initialize all kernels + derivs used in jacobi pitd fits
        self.texpKernel=texpKernel(self.comp)

        
        # List to contain all PDF objects to plot
        self.pdfs2Plot=[(self.pdfLT,self.pdfLTErr,r'$%s_{q_{\rm %s}/N}\left(x,\mu^2\right)^{\left[%s\right]}%s$'%(self.pdfVariety,self.name,self.truncStr,self.pdfNameSuffix)),\
                        (self.pdfAZ,self.pdfAZErr,r'$\mathcal{O}(a/\left|z\right|)$')]
        for n,h in enumerate(self.HT):
            # No longer want twist-6/twist-8 pieces, so just truncate the enumeration to keep functionality
            if n < 2:
                self.pdfs2Plot.append( (h.pdf,h.pdfErr,r'$\mathcal{O}(z^%i\Lambda_{\rm QCD}^%i)$'%(2*h.pitdCorrPower,2*h.pitdCorrPower)) )

            

    # Parse the fit parameters
    def parse(self):
        # Form the ensemble avg of each fit parameter, by dividing by number of gauge configs
        for k, v in self.fitParamsAll.items():
            # Sum all params, skipping 'norm' if it is None
            dum=0.0
            if v is None:
                continue
            else:
                for n in v:
                    dum+=n
            dum/=self.cfgs
            self.pAvg.update({k: dum})

        # Form parameter covariance matrix + its easy matview (pCovView)
        self.pCovView=np.zeros((len(self.pAvg),len(self.pAvg)))
        for ni, ki in enumerate(self.pAvg.keys()):
            for nj, kj in enumerate(self.pAvg.keys()):
                key=(ki,kj)
                dum=0.0
                for g in range(0,self.cfgs):
                    dum+=(self.fitParamsAll[ki][g]-self.pAvg[ki])*\
                          (self.fitParamsAll[kj][g]-self.pAvg[kj])
                dum*=((1.0*(self.cfgs-1))/self.cfgs)

                if not(self.correlated) and ki != kj:
                    self.pCov.update({key: 0.0})
                    self.pCovView[ni,nj]=0.0
                else:
                    self.pCov.update({key: dum})
                    self.pCovView[ni,nj]=dum

        # Lastly normalize pCovView to diagonal entries
        diag=[self.pCovView[d,d] for d in range(0,len(self.pCovView))]
        for i in range(0,len(self.pCovView)):
            for j in range(0,len(self.pCovView)):
                self.pCovView[i,j]/=np.sqrt(diag[i]*diag[j])


    # Plot Fit Param Covariance heatmap
    def plotCovHMap(self,fig,ax,cmap):
        # First set colormap based on some dummy array w/ values \in [-1,1]
        pCovViewDum=[[-1.0,1.0],[-0.25,0.5]]
        daxDum=ax.matshow(pCovViewDum,cmap=cmap)
        cbar = fig.colorbar(daxDum)
        ticks=[-1.0,-0.5,0.0,0.5,1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([r'${}$'.format(tikval) for tikval in ticks])

        print("pcov before")
        print(self.pCovView[4:,4:])
        
        # Now plot the actual parameter covariance
        dax=ax.matshow(self.pCovView[4:,4:],cmap=cmap)
        ax.set_xticks(np.arange(len(self.pAvg)-4))
        ax.set_yticks(np.arange(len(self.pAvg)-4))
        # ax.set_yticklabels('$%s$'%k for k in self.pAvg.keys() if k is not 'chi2')
        # ax.set_xticklabels('$%s$'%k for k in self.pAvg.keys() if k is not 'chi2')
        ax.set_yticklabels('$%s$'%k for k in self.pAvg.keys() if k not in ('L2','chi2','L2/dof','chi2/dof'))
        ax.set_xticklabels('$%s$'%k for k in self.pAvg.keys() if k not in ('L2','chi2','L2/dof','chi2/dof'))


        print("pcov after")
        print(self.pCovView[4:,4:])
        
        # Add correlation coefficient inside each cell
        for ni in range(0,len(self.pCovView[4:,4:])):
            for nj in range(0,len(self.pCovView[4:,4:])):
                ax.text(ni,nj,s=r'$%.2f$'%self.pCovView[4:,4:][ni,nj],\
                        ha='center',va='center',fontsize=14)


    # Print out fit parameters determined from jackknife
    def printFitParams(self):
        for k in self.pAvg.keys():
            print("     %s (%s) = %.7f +/- %.7f"%(k,self.name,self.pAvg[k],\
                                                  np.sqrt(self.pCov[(k,k)])))

    # Plot various pdf fits
    def plotPDFs(self,ax,insetTruth=False):
        # Optionally suppress plotting jacobi corrections in inset plot
        if not insetTruth:
            # for n, t in enumerate([(self.pdfLT,self.pdfLTErr,\
            #                         r'$f_{q_{\rm %s}/N}\left(x,\mu^2\right)^{\left[%s\right]}_{%s}$'%\
            #                         (self.name,self.truncStr,self.kernel)),\
            #                        (self.pdfAZ,self.pdfAZErr,r'$\mathcal{O}(a/\left|z\right|)_{%s}$'%\
            #                         self.kernel),\
            #                        (self.pdfT4,self.pdfT4Err,\
            #                         r'$\mathcal{O}(z^2\Lambda_{\rm QCD}^2)_{%s}$'%self.kernel),\
            #                        (self.pdfT6,self.pdfT6Err,\
            #                         r'$\mathcal{O}(z^4\Lambda_{\rm QCD}^4)_{%s}$'%self.kernel)]):
            for n, t in enumerate(self.pdfs2Plot):
                ax.plot(self.x,t[0](self.x),color=self.cols[n],label=t[2])
                ax.fill_between(self.x,t[0](self.x)+t[1](self.x)[0],\
                                t[0](self.x)-t[1](self.x)[0],color=self.cols[n],alpha=0.3,\
                                label=t[2],lw=0)
        else:
            for n, t in enumerate([(self.pdfLT,self.pdfLTErr,\
                                    r'$%s_{q_{\rm %s}/N}\left(x,\mu^2\right)^{\left[%s\right]}$'%\
                                    (self.pdfVariety,self.name,self.truncStr))]):
                ax.plot(self.x,t[0](self.x),color=self.cols[n],label=t[2])
                ax.fill_between(self.x,t[0](self.x)+t[1](self.x)[0],\
                                t[0](self.x)-t[1](self.x)[0],color=self.cols[n],alpha=0.3,\
                                label=t[2],lw=0)
            
    # Plot various x*pdf fits
    def plotXPDFs(self,ax):
        # for n, t in enumerate([(self.pdfLT,self.pdfLTErr,\
        #                         r'$xf_{q_{\rm %s}/N}\left(x,\mu^2\right)^{\left[%s\right]}_{%s}$'%\
        #                         (self.name,self.truncStr,self.kernel)),\
        #                        (self.pdfAZ,self.pdfAZErr,r'$\mathcal{O}(a/\left|z\right|)_{%s}$'\
        #                         %self.kernel),
        #                        (self.pdfT4,self.pdfT4Err,\
        #                         r'$\mathcal{O}(z^2\Lambda_{\rm QCD}^2)_{%s}$'%self.kernel),\
        #                        (self.pdfT6,self.pdfT6Err,\
        #                         r'$\mathcal{O}(z^4\Lambda_{\rm QCD}^4)_{%s}$'%self.kernel)]):
        for n, t in enumerate(self.pdfs2Plot):
            ax.plot(self.x,self.x*t[0](self.x),color=self.cols[n],label=t[2])
            ax.fill_between(self.x,self.x*t[0](self.x)+self.x*t[1](self.x)[0],\
                            self.x*t[0](self.x)-self.x*t[1](self.x)[0],color=self.cols[n],\
                            alpha=0.3,label=t[2],lw=0)

    # Plot derived ITDs
    def plotITDs(self,ax):
        ax.plot(self.nu,self.itdLT(self.nu),color=self.cols[0],\
                label=r'$\mathfrak{%s}\ Q\left(\nu,\mu^2\right)^{\left[%s\right]}_{%s}$'\
                %(self.comp,self.truncStr,self.kernel),alpha=0.3)
        ax.fill_between(self.nu,[x+y for x, y in \
                                 zip(self.itdLT(self.nu),self.itdLTErr(self.nu))],\
                        [x-y for x, y in zip(self.itdLT(self.nu),self.itdLTErr(self.nu))],\
                        color=self.cols[0],\
                        label=r'$\mathfrak{%s}\ Q\left(\nu,\mu^2\right)^{\left[%s\right]}_{%s}$'\
                        %(self.comp,self.truncStr,self.kernel),alpha=0.3,lw=0)
        # Add \chi2 result to ITD plot, although the fit w/ jacobi polynomials was performed on the pseudo-ITDs!
        # xRange=abs(ax.get_xlim()[1]-ax.get_xlim()[0])
        # yRange=abs(ax.get_ylim()[1]-ax.get_ylim()[0])
        # ax.text(ax.get_xlim()[0]+0.25*xRange,ax.get_ylim()+0.15*yRange,\
        #         r'$\chi_r^2\left[%s\right]=%.5f$'%(self.kernel,self.pAvg['chi2']),fontsize=18)
        # ax.text(2.5,-0.15,\
        #         r'$\chi_r^2\left[%s\right]\simeq%.5f$'%(self.kernel,self.pAvg['chi2']),fontsize=18)



    # LEADING TWIST PDF
    def pdfLT(self,x):
        ltSum=0.0
        for n in range(0,self.numLT):
            ltSum += self.pAvg['C^{lt}_'+str(n)]*Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
        return np.power(x,self.pAvg['\\alpha'])*np.power(1-x,self.pAvg['\\beta'])*ltSum
    # LEADING TWIST PDF ERRORS
    def pdfLTErr(self,x):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for l in range(0,self.numLT):
            partials.update({'C^{lt}_%d'%l: 0.0})
        partials['\\alpha'] += np.log(x)*self.pdfLT(x)
        partials['\\beta'] += np.log(1-x)*self.pdfLT(x)
        
        for n in range(0,self.numLT):
            partials['\\alpha'] += np.power(x,self.pAvg['\\alpha'])*\
                                   np.power(1-x,self.pAvg['\\beta'])*\
                                   self.pAvg['C^{lt}_'+str(n)]*\
                                   dAlphaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['\\beta'] += np.power(x,self.pAvg['\\alpha'])*\
                                  np.power(1-x,self.pAvg['\\beta'])*\
                                  self.pAvg['C^{lt}_'+str(n)]*\
                                  dBetaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['C^{lt}_'+str(n)] = np.power(x,self.pAvg['\\alpha'])*\
                                         np.power(1-x,self.pAvg['\\beta'])*\
                                         Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
        # Propagate error
        # and return the total error & partials dict (used for ITD error derivation)
        return propError(self.pCov,partials), partials
    
    # A/Z CORRECTIONS PDF
    def pdfAZ(self,x):
        azSum=0.0
        for n in range(self.cs,self.numAZ+self.cs):
            azSum += self.pAvg['C^{az}_'+str(n)]*Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
        return np.power(x,self.pAvg['\\alpha'])*np.power(1-x,self.pAvg['\\beta'])*azSum
    # A/Z CORRECTIONS PDF ERRORS
    def pdfAZErr(self,x):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for a in range(self.cs,self.numAZ+self.cs):
            partials.update({'C^{az}_%d'%a: 0.0})
        partials['\\alpha'] += np.log(x)*self.pdfAZ(x)
        partials['\\beta'] += np.log(1-x)*self.pdfAZ(x)
        
        for n in range(self.cs,self.numAZ+self.cs):
            partials['\\alpha'] += np.power(x,self.pAvg['\\alpha'])*\
                                   np.power(1-x,self.pAvg['\\beta'])*\
                                   self.pAvg['C^{az}_'+str(n)]*\
                                   dAlphaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['\\beta'] += np.power(x,self.pAvg['\\alpha'])*\
                                  np.power(1-x,self.pAvg['\\beta'])*\
                                  self.pAvg['C^{az}_'+str(n)]*\
                                  dBetaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
            partials['C^{az}_'+str(n)] += np.power(x,self.pAvg['\\alpha'])*\
                                          np.power(1-x,self.pAvg['\\beta'])*\
                                          Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
        # Propagate error and return
        # and return the total error & partials dict (used for ITD error derivation)
        return propError(self.pCov,partials), partials


    # # Higher-twist PDF
    # def pdfHT(self,x):
    #     totHT=0.0
    #     for n,h in self.numHTs:
    #         ht=jacobiHT(2*(n+2),h,self.cs,self.pAvg,self.pCov)
    #         totHT+=ht.pdf(x)
    #     return totHT

    # # Higher-twist PDF Error
    # def pdfHTErr(self,x):
    #     totHT=0.0
    #     for n,h in self.numHTs:
    #         ht=jacobiHT(2*(n+2),h,self.cs,self.pAvg,self.pCov)
    #         totHT+=ht.pdfErr(x)
    #     return totHT


    
    # # TWIST-4 PDF
    # def pdfT4(self,x):
    #     t4Sum=0.0
    #     for n in range(self.cs,self.numT4+self.cs):
    #         t4Sum += self.pAvg['C^{t4}_'+str(n)]*Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #     return np.power(x,self.pAvg['\\alpha'])*np.power(1-x,self.pAvg['\\beta'])*t4Sum
    # # TWIST-4 PDF + ERRORS
    # def pdfT4Err(self,x):
    #     partials={'\\alpha': 0.0, '\\beta': 0.0}
    #     for t in range(self.cs,self.numT4+self.cs):
    #         partials.update({'C^{t4}_%d'%t: 0.0})
    #     partials['\\alpha'] += np.log(x)*self.pdfT4(x)
    #     partials['\\beta'] += np.log(1-x)*self.pdfT4(x)
        
    #     for n in range(self.cs,self.numT4+self.cs):
    #         partials['\\alpha'] += np.power(x,self.pAvg['\\alpha'])*\
    #                                np.power(1-x,self.pAvg['\\beta'])*\
    #                                self.pAvg['C^{t4}_'+str(n)]*\
    #                                dAlphaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #         partials['\\beta'] += np.power(x,self.pAvg['\\alpha'])*\
    #                               np.power(1-x,self.pAvg['\\beta'])*\
    #                               self.pAvg['C^{t4}_'+str(n)]*\
    #                               dBetaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #         partials['C^{t4}_'+str(n)] += np.power(x,self.pAvg['\\alpha'])*\
    #                                       np.power(1-x,self.pAvg['\\beta'])*\
    #                                       Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #     # Propagate error and return
    #     # and return the total error & partials dict (used for ITD error derivation)
    #     return propError(self.pCov,partials), partials

    # # TWIST-6 PDF
    # def pdfT6(self,x):
    #     t6Sum=0.0
    #     for n in range(self.cs,self.numT6+self.cs):
    #         t6Sum += self.pAvg['C^{t6}_'+str(n)]*Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #     return np.power(x,self.pAvg['\\alpha'])*np.power(1-x,self.pAvg['\\beta'])*t6Sum
    # # TWIST-6 PDF + ERRORS
    # def pdfT6Err(self,x):
    #     partials={'\\alpha': 0.0, '\\beta': 0.0}
    #     for t in range(self.cs,self.numT6+self.cs):
    #         partials.update({'C^{t6}_%d'%t: 0.0})
    #     partials['\\alpha'] += np.log(x)*self.pdfT6(x)
    #     partials['\\beta'] += np.log(1-x)*self.pdfT6(x)
        
    #     for n in range(self.cs,self.numT6+self.cs):
    #         partials['\\alpha'] += np.power(x,self.pAvg['\\alpha'])*\
    #                                np.power(1-x,self.pAvg['\\beta'])*\
    #                                self.pAvg['C^{t6}_'+str(n)]*\
    #                                dAlphaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #         partials['\\beta'] += np.power(x,self.pAvg['\\alpha'])*\
    #                               np.power(1-x,self.pAvg['\\beta'])*\
    #                               self.pAvg['C^{t6}_'+str(n)]*\
    #                               dBetaJacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #         partials['C^{t6}_'+str(n)] += np.power(x,self.pAvg['\\alpha'])*\
    #                                       np.power(1-x,self.pAvg['\\beta'])*\
    #                                       Jacobi(n,self.pAvg['\\alpha'],self.pAvg['\\beta'],x)
    #     # Propagate error and return
    #     # and return the total error & partials dict (used for ITD error derivation)
    #     return propError(self.pCov,partials), partials

    #--------------------------------------------------
    # Leading-Twist pitd for some z (Continuous in \nu)
    #--------------------------------------------------
    def lt_pitd(self,nu,z):
        summ=0.0
        for n in range(0,self.numLT):
            summ+=self.pAvg['C^{lt}_'+str(n)]*\
                   self.texpKernel.NLO(n,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu,z,\
                                       dirac=self.dirac)
        return summ

    #--------------------------------------------------------
    # Leading-Twist pitd Error for some z (Continuous in \nu)
    #--------------------------------------------------------
    def lt_pitd_error(self,nu,z):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for l in range(0,self.numLT):
            partials.update({'C^{lt}_%d'%l: 0.0})
            
        for n in range(0,self.numLT):
            partials['\\alpha'] += self.texpKernel.dA_NLO(n, 75, self.pAvg['\\alpha'],\
                                                          self.pAvg['\\beta'], nu, z,\
                                                          dirac=self.dirac)*\
                                                          self.pAvg['C^{lt}_'+str(n)]
            partials['\\beta'] += self.texpKernel.dB_NLO(n, 75, self.pAvg['\\alpha'],\
                                                         self.pAvg['\\beta'], nu, z,\
                                                         dirac=self.dirac)*\
                                                         self.pAvg['C^{lt}_'+str(n)]
            partials['C^{lt}_'+str(n)] += self.texpKernel.NLO(n, 75, self.pAvg['\\alpha'],\
                                                              self.pAvg['\\beta'], nu, z,\
                                                              dirac=self.dirac)
        # Propagate error and return
        return propError(self.pCov,partials)
        

    #--------------------------------------------------------
    # (a/z)^(self.texpKernel.subleadPow) corrections pitd for some z (Continuous in \nu)
    #--------------------------------------------------------
    def az_pitd(self,nu,z):
        azSum=0.0
        if self.numAZ > 0:
            # ##############################################################
            # # 5/18/2021: [EXP] Explanation for conditional correction sums
            # ##############################################################
            # # Sum starting at n=1 is set for Real pITD
            # # az params are stored in dict at C^{az}_1,...
            # # So 'n' passed to Tree(n,75,...) must be n-1 for Imag
            # for n in range(1,self.numAZ+1): #self.texpKernel.subleadPow
            #     nn=n
            #     if self.comp == 'Im':
            #         nn=n-1
            #     azSum+=np.power(1.0/z,self.texpKernel.subleadPow)*self.pAvg['C^{az}_'+str(n)]*\
            #             self.texpKernel.Tree(nn,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)

            for n in range(self.cs,self.numAZ+self.cs): #self.texpKernel.subleadPow
                azSum+=np.power(1.0/z,self.texpKernel.subleadPow)*self.pAvg['C^{az}_'+str(n)]*\
                        self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
        return azSum

    #--------------------------------------------------------------
    # (a/z)^(self.texpKernel.subleadPow) corrections pitd Error for some z (Continuous in \nu)
    #--------------------------------------------------------------
    def az_pitd_error(self,nu,z):
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for a in range(self.cs,self.numAZ+self.cs):
            partials.update({'C^{az}_%d'%a: 0.0})

        if self.numAZ > 0:
            # for n in range(1,self.numAZ+1):
            #     ### See [EXP] explanation above
            #     nn=n
            #     if self.comp == 'Im':
            #         nn=n-1

            for n in range(self.cs,self.numAZ+self.cs):        
                partials['\\alpha'] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                                       self.pAvg['C^{az}_'+str(n)]*\
                                       self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
                                                               self.pAvg['\\beta'],nu)
                partials['\\beta'] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                                      self.pAvg['C^{az}_'+str(n)]*\
                                      self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
                                                              self.pAvg['\\beta'],nu)
                partials['C^{az}_'+str(n)] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                                              self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
                                                                   self.pAvg['\\beta'],nu)
        # Propagate error and return
        return propError(self.pCov,partials)

    # #------------------------------------------------------------------------
    # # Twist-4 (z * \Lambda)^2 corrections pitd for some z (Continuous in \nu)
    # #------------------------------------------------------------------------
    # def t4_pitd(self,nu,z):
    #     t4Sum=0.0
    #     if self.numT4 > 0:
    #         # for n in range(1,self.numT4+1):
    #         #     ### See [EXP] explanation above
    #         #     nn=n
    #         #     if self.comp == 'Im':
    #         #         nn=n-1
    #         for n in range(self.cs,self.numT4+self.cs):
    #             t4Sum+=np.power(z*LambdaQCD,2)*self.pAvg['C^{t4}_'+str(n)]*\
    #                     self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
    #     return t4Sum

    # #------------------------------------------------------------------------------
    # # Twist-4 (z * \Lambda)^2 corrections pitd Error for some z (Continuous in \nu)
    # #------------------------------------------------------------------------------
    # def t4_pitd_error(self,nu,z):
    #     partials={'\\alpha': 0.0, '\\beta': 0.0}
    #     for t in range(self.cs,self.numT4+self.cs):
    #         partials.update({'C^{t4}_%d'%t: 0.0})

    #     if self.numT4 > 0:
    #         # for n in range(1,self.numT4+1):
    #         #     ### See [EXP] explanation above
    #         #     nn=n
    #         #     if self.comp == 'Im':
    #         #         nn=n-1
    #         for n in range(self.cs,self.numT4+self.cs):
    #             partials['\\alpha'] += np.power(z*LambdaQCD,2)*self.pAvg['C^{t4}_'+str(n)]*\
    #                                    self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
    #                                                            self.pAvg['\\beta'],nu)
    #             partials['\\beta'] += np.power(z*LambdaQCD,2)*self.pAvg['C^{t4}_'+str(n)]*\
    #                                   self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
    #                                                           self.pAvg['\\beta'],nu)
    #             partials['C^{t4}_'+str(n)] = np.power(z*LambdaQCD,2)*\
    #                                          self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
    #                                                               self.pAvg['\\beta'],nu)
    #     # Propagate error and return
    #     return propError(self.pCov,partials)

    # #------------------------------------------------------------------------
    # # Twist-6 (z * \Lambda)^4 corrections pitd for some z (Continuous in \nu)
    # #------------------------------------------------------------------------
    # def t6_pitd(self,nu,z):
    #     t6Sum=0.0
    #     if self.numT6 > 0:
    #         # for n in range(1,self.numT6+1):
    #         #     ### See [EXP] explanation above
    #         #     nn=n
    #         #     if self.comp == 'Im':
    #         #         nn=n-1
    #         for n in range(self.cs,self.numT6+self.cs):
    #             t6Sum+=np.power(z*LambdaQCD,4)*self.pAvg['C^{t6}_'+str(n)]*\
    #                     self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],self.pAvg['\\beta'],nu)
    #     return t6Sum

    # #------------------------------------------------------------------------------
    # # Twist-6 (z * \Lambda)^4 corrections pitd Error for some z (Continuous in \nu)
    # #------------------------------------------------------------------------------
    # def t6_pitd_error(self,nu,z):
    #     partials={'\\alpha': 0.0, '\\beta': 0.0}
    #     for t in range(self.cs,self.numT6+self.cs):
    #         partials.update({'C^{t6}_%d'%t: 0.0})

    #     if self.numT6 > 0:
    #         # for n in range(1,self.numT6+1):
    #         #     ### See [EXP] explanation above
    #         #     nn=n
    #         #     if self.comp == 'Im':
    #         #         nn=n-1
    #         for n in range(self.cs,self.numT6+self.cs):
    #             partials['\\alpha'] += np.power(z*LambdaQCD,4)*self.pAvg['C^{t6}_'+str(n)]*\
    #                                    self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
    #                                                            self.pAvg['\\beta'],nu)
    #             partials['\\beta'] += np.power(z*LambdaQCD,4)*self.pAvg['C^{t6}_'+str(n)]*\
    #                                   self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
    #                                                           self.pAvg['\\beta'],nu)
    #             partials['C^{t6}_'+str(n)] = np.power(z*LambdaQCD,4)*\
    #                                          self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
    #                                                               self.pAvg['\\beta'],nu)
    #     # Propagate error and return
    #     return propError(self.pCov,partials)

    #---------------------------------------------------------------------------
    # The pITD predicted by model and corrections for some z (Continuous in \nu)
    #---------------------------------------------------------------------------
    def pitd(self,nu,z):
        # return self.lt_pitd(nu,z)+self.az_pitd(nu,z)+self.t4_pitd(nu,z)+self.t6_pitd(nu,z)
        totPITD=self.lt_pitd(nu,z)+self.az_pitd(nu,z)
        for h in self.HT:
            totPITD+=h.pitd(nu,z)

        return totPITD
        
    #---------------------------------------------------------------------------------
    # The pITD Error Band predicted by model and corrections for some z (Cont. in \nu)
    #---------------------------------------------------------------------------------
    def pitdError(self,nu,z):
        # return self.lt_pitd_error(nu,z)+self.az_pitd_error(nu,z)+self.t4_pitd_error(nu,z)\
        #     +self.t6_pitd_error(nu,z)
        partials={'\\alpha': 0.0, '\\beta': 0.0}
        for l in range(0,self.numLT):
            partials.update({'C^{lt}_%d'%l: 0.0})
        for a in range(self.cs,self.numAZ+self.cs):
            partials.update({'C^{az}_%d'%a: 0.0})
        # for t in range(self.cs,self.numT4+self.cs):
        #     partials.update({'C^{t4}_%d'%t: 0.0})
        # for t in range(self.cs,self.numT6+self.cs):
        #     partials.update({'C^{t6}_%d'%t: 0.0})

        for n in range(0,self.numLT):
            partials['\\alpha'] += self.texpKernel.dA_NLO(n, 75, self.pAvg['\\alpha'],\
                                                          self.pAvg['\\beta'], nu, z,\
                                                          dirac=self.dirac)*\
                                                          self.pAvg['C^{lt}_'+str(n)]
            partials['\\beta'] += self.texpKernel.dB_NLO(n, 75, self.pAvg['\\alpha'],\
                                                         self.pAvg['\\beta'], nu, z,\
                                                         dirac=self.dirac)*\
                                                         self.pAvg['C^{lt}_'+str(n)]
            partials['C^{lt}_'+str(n)] += self.texpKernel.NLO(n, 75, self.pAvg['\\alpha'],\
                                                              self.pAvg['\\beta'], nu, z,\
                                                              dirac=self.dirac)
            
        for n in range(self.cs,self.numAZ+self.cs):        
            partials['\\alpha'] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                self.pAvg['C^{az}_'+str(n)]*\
                self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
                                        self.pAvg['\\beta'],nu)
            partials['\\beta'] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                self.pAvg['C^{az}_'+str(n)]*\
                self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
                                        self.pAvg['\\beta'],nu)
            partials['C^{az}_'+str(n)] += np.power(1.0/z,self.texpKernel.subleadPow)*\
                self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
                                     self.pAvg['\\beta'],nu)

        for h in self.HT:
            h.pitd_error(partials,nu,z)

        # for n in range(self.cs,self.numT4+self.cs):
        #     partials['\\alpha'] += np.power(z*LambdaQCD,2)*self.pAvg['C^{t4}_'+str(n)]*\
        #         self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
        #                                 self.pAvg['\\beta'],nu)
        #     partials['\\beta'] += np.power(z*LambdaQCD,2)*self.pAvg['C^{t4}_'+str(n)]*\
        #         self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
        #                                 self.pAvg['\\beta'],nu)
        #     partials['C^{t4}_'+str(n)] = np.power(z*LambdaQCD,2)*\
        #         self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
        #                              self.pAvg['\\beta'],nu)

        # for n in range(self.cs,self.numT6+self.cs):
        #     partials['\\alpha'] += np.power(z*LambdaQCD,4)*self.pAvg['C^{t6}_'+str(n)]*\
        #         self.texpKernel.dA_Tree(n,75,self.pAvg['\\alpha'],\
        #                                 self.pAvg['\\beta'],nu)
        #     partials['\\beta'] += np.power(z*LambdaQCD,4)*self.pAvg['C^{t6}_'+str(n)]*\
        #         self.texpKernel.dB_Tree(n,75,self.pAvg['\\alpha'],\
        #                                 self.pAvg['\\beta'],nu)
        #     partials['C^{t6}_'+str(n)] = np.power(z*LambdaQCD,4)*\
        #         self.texpKernel.Tree(n,75,self.pAvg['\\alpha'],\
        #                              self.pAvg['\\beta'],nu)

        return propError(self.pCov,partials)

    # Plot pITD w/ No Error
    def plotPITDNoErr(self,ax,z,col,hatch=None):
        ax.plot(self.nu,self.pitd(self.nu,z),color=col,\
                label=r'$\mathfrak{%s}\ \mathfrak{M}(\nu,z^2)$'%self.comp)

    # Plot pITD
    def plotPITD(self,ax,z,col,hatch=None):
        ax.plot(self.nu,self.pitd(self.nu,z),color=col,\
                label=r'$\mathfrak{%s}\ \mathfrak{M}(\nu,z^2)$'%self.comp)
        ax.fill_between(self.nu,[x+y for x, y in\
                                 zip(self.pitd(self.nu,z),self.pitdError(self.nu,z))],\
                        [x-y for x, y in zip(self.pitd(self.nu,z),self.pitdError(self.nu,z))],\
                        color=col,alpha=0.3,\
                        label=r'$\mathfrak{%s}\ \mathfrak{M}(\nu,z^2)$'%self.comp,lw=0,hatch=hatch)

    #-------------------------------------------------------------
    # Derived Leading-Twist ITD from Jacobi Polynomial rep. of PDF
    #-------------------------------------------------------------
    def itdLT(self,nu):
        itd = []
        s = None
        for i_nu in nu:
            if self.comp == 'Re':
                s = lambda x: np.cos(i_nu*x)*self.pdfLT(x)
            if self.comp == 'Im':
                s = lambda x: np.sin(i_nu*x)*self.pdfLT(x)
            itd.append(integrate.quad(s,0,1)[0])
        return itd
    #----------------------------------------------------------------------
    # Error of Derived Leading-Twist ITD from Jacobi Polynomial rep. of PDF
    #----------------------------------------------------------------------
    def itdLTErr(self,nu):
        err_itd = []
        s = None
        for i_nu in nu:
            partials={'\\alpha': 0.0, '\\beta': 0.0}
            for l in range(0,self.numLT):
                partials.update({'C^{lt}_%d'%l: 0.0})

            for k,v in partials.items():
                if self.comp == 'Re':
                    s = lambda x: np.cos(i_nu*x)*self.pdfLTErr(x)[1][k]
                if self.comp == 'Im':
                    s = lambda x: np.sin(i_nu*x)*self.pdfLTErr(x)[1][k]
                # Update the partials
                partials.update({k: integrate.quad(s,0,1)[0]})

            # Propagate error and return
            err_itd.append(propError(self.pCov,partials))
        return err_itd

    
    
############################################################
##### UTILITIES TO PLOT THE CONTENTS OF PASSED H5 FILE #####
############################################################
def h5Plot(h5file,zcut,pcut,realAxes,imagAxes,cfgs=0,dtypeName='',dispCMap=[],\
           dirac=8,oldH5Hierarchy=False,sysErrH5File=None,\
           showZSepOnPlot=False,plotSingleZ=False):
    h5In = h5py.File(h5file,'r')
    h5InSys = None if sysErrH5File == None else h5py.File(sysErrH5File,'r')
    print("Attempting to read from h5 files:\n\t\t%s\n\t\t%s"%(h5file,sysErrH5File))

    allZ=np.arange(1,9)
    allP=np.arange(1,7)

    pmin, pmax = [int(p) for p in pcut.split('.')]
    zmin, zmax = [int(z) for z in zcut.split('.')]

    if plotSingleZ:
        allZ=[plotSingleZ]
    
    # if type(zcut) is str:
    #     print("zcut is a str")
    #     zmin, zmax = [int(z) for z in zcut.split('.')]
    # elif type(zcut) is int:
    #     print("zcut is an int")
    #     zmin, zmax = zcut, zcut
    # else:
    #     print("Don't know what kind of zsep you've asked me to plot from h5")
    #     sys.exit()
    
    # Z=[]
    # if type(zcut) == str:
    #     zmin = int(zcut.split('.')[0]); zmax = int(zcut.split('.')[1]);
    #     Z = [zi for zi in range(zmin,zmax+1)]
    # elif type(zcut) == int:
    #     Z.append(zcut)
    # else:
    #     # print "Don't know what kind of zsep you've asked me to plot from h5"
    #     print("Don't know what kind of zsep you've asked me to plot from h5")
    #     sys.exit()



 
    # pmin = int(pcut.split('.')[0]); pmax = int(pcut.split('.')[1]);
    # P = [pi for pi in range(pmin,pmax+1)]


    # A sloppy handle for data covariance heatmaps
    DatHeat=[]
    
    for comp in ["Re", "Im"]:
        # Dictionary to hold avg matelem for each z/p combo (to easily construct covHeatMap)
        avgMatDict={}
        matDict={}

        for z in allZ:
        # for z in Z:
            # Set ztag based on hierarchy version of h5
            ztag="zsep%d"%z if oldH5Hierarchy else "zsep00%d"%z

            for m in allP:
            # for m in range(pmin,pmax+1):
                # Set ptag based on hierarchy version of h5
                ptag="pz%d"%m if oldH5Hierarchy else "pf00%d_pi00%d"%(m,m)


                thisPZKey=("z/a=%d"%z,"p=%d"%m)
                # Make an empty array for this z/m key in matDict
                matDict.update({thisPZKey: []})


                THISCOLOR=dispCMap[z]
                print("CHECKS: (z,p) = (%i,%i)  with zmin = %i, zmax = %i, pmin = %i, pmax = %i"\
                      %(z,m,zmin,zmax,pmin,pmax))
                if z < zmin or z > zmax or m < pmin or m > pmax:
                    THISCOLOR='gray'

                if THISCOLOR == 'gray':
                    print("     This color = %s"%str(THISCOLOR))

                
                ioffeTime, avgMat, avgMatErr, avgMatSys, avgMatErrSys = (-1,0.0,0.0,0.0,0.0)
                for g in range(0,cfgs):
                    ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                          (insertions[dirac]['Redstar'],ztag,ptag,\
                                           comp,dtypeName)][g]
                    # Collect matelem fits from h5 used to set matelem sys. error
                    try:
                        ioffeTime, matSys = h5InSys['/%s/%s/%s/jack/%s/%s'%\
                                                    (insertions[dirac]['Redstar'],ztag,ptag,\
                                                     comp,dtypeName)][g]
                    except:
                        matSys = mat
                    #------------------

                    avgMat += mat
                    avgMatSys += matSys
                    matDict[thisPZKey].append(mat)
    
                avgMat *= (1.0/cfgs)
                avgMatSys *= (1.0/cfgs)

                # Pack avgMatDict
                avgMatDict.update({thisPZKey: avgMat})

                
                # Determine the error (and potentially a matelem sys error if h5InSys is passed)
                for g in range(0,cfgs):
                    ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                          (insertions[dirac]['Redstar'],ztag,ptag,\
                                           comp,dtypeName)][g]
                        
                    avgMatErr += np.power( mat - avgMat, 2)


                # Since only variance is plotted w/in this method, squared diff. btwn. avg matelems from h5In & h5InSys is added to avgMatErr
                avgMatErrSys += avgMatErr + np.power( avgMatSys - avgMat, 2)
                avgMatErrSys = np.sqrt( ((1.0*(cfgs-1))/cfgs)*avgMatErrSys )
                
                avgMatErr = np.sqrt( ((1.0*(cfgs-1))/cfgs)*avgMatErr )

                
                if comp == "Re":
                    for a in realAxes:
                        if a != None:
                            # Plot error of fitted dataset, potentially together w/ systematic error
                            for errPair in [(avgMatErrSys, 0.6), (avgMatErr, 1.0)]:
                                a.errorbar(ioffeTime, avgMat, yerr=errPair[0], fmt='o',\
                                           color=THISCOLOR,mec=THISCOLOR,\
                                           mfc=THISCOLOR,label=r'$z=%s$'%z,alpha=errPair[1])
                            a.axhline(y=0.0,ls='--',dashes=(5,15),color='gray',lw=0.5)
                                
                            # Add displacement text to this subaxes
                            if showZSepOnPlot:
                                xmin, xmax = a.get_xlim()
                                ymin, ymax = a.get_ylim()
                                a.text(xmin+0.1*abs(xmax-xmin),ymin+0.1*abs(ymax-ymin),\
                                       r'$z/a=%d$'%z,fontsize=22)
                if comp == "Im":
                    for a in imagAxes:
                        if a != None:
                            # Plot error of fitted dataset, potentially together w/ systematic error
                            for errPair in [(avgMatErrSys, 0.6), (avgMatErr, 1.0)]:
                                a.errorbar(ioffeTime, avgMat, yerr=errPair[0], fmt='o',\
                                           color=THISCOLOR,mec=THISCOLOR,\
                                           mfc=THISCOLOR,label=r'$z=%s$'%z,alpha=errPair[1])
                            a.axhline(y=0.0,ls='--',dashes=(5,15),color='gray',lw=0.5)

                            # Add displacement text to this subaxes
                            if showZSepOnPlot:
                                xmin, xmax = a.get_xlim()
                                ymin, ymax = a.get_ylim()
                                a.text(xmin+0.1*abs(xmax-xmin),ymin+0.9*abs(ymax-ymin),\
                                       r'$z/a=%d$'%z,fontsize=22)
