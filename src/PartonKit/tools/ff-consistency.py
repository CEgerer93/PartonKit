#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys,optparse
import h5py

sys.path.append("/home/colin/QCD/pseudoDists/strFuncViz")
sys.path.append("/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo")
import common_fig
import pitd_util


# Vector renormalization
ZV=0.84


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-f", "--ampH5File", type="str", default="",
                  help='H5 file containing amplitudes (default = '')')
parser.add_option("-t", "--twoPtH5File", type="str", default="",
                  help='H5 file containing 2pt fitted energies (default = '')')
parser.add_option("--selectPfPi", type="str", default="",
                  help='Select specific pf/pi combo to access (default = '')')
parser.add_option("-c", "--component", type="str", default="real",
                  help='Component of amplitudes to plot (default = "real")')

# Parse the input arguments
(options, args) = parser.parse_args()


comp=options.component


figGM=plt.figure(figsize=(8,10)); axGM=figGM.gca();
axGM.set_xlabel(r'$Q^2\ \ \left({\rm GeV}^2\right)$')
axGM.set_ylabel(r'$G_M\left(Q^2\right)$')
axGM.set_xlim([0,1.3])
figGE=plt.figure(figsize=(8,10)); axGE=figGE.gca();
axGE.set_xlabel(r'$Q^2\ \ \left({\rm GeV}^2\right)$')
axGE.set_ylabel(r'$G_E\left(Q^2\right)$')
axGE.set_xlim([0,1.3])


##########################################################################################
# Sungwoo FF stuff
##########################################################################################
# Momentum transfer squared - strategy {4} of Table XVII in PRD.105.054505
qvecToQ2={ 'a094m270' : { (1,0,0): 0.164,
                          (1,1,0): 0.314,
                          (1,1,1): 0.453,
                          (2,0,0): 0.598,
                          (2,1,0): 0.716,
                          (2,1,1): 0.839,
                          (2,2,0): 1.046,
                          (2,2,1): 1.172,
                          (3,0,0): 1.186,
                          (3,1,0): 1.293 },
           'a094m270L': { (1,0,0): 0.074,
                          (1,1,0): 0.146,
                          (1,1,1): 0.215,
                          (2,0,0): 0.281,
                          (2,1,0): 0.346,
                          (2,1,1): 0.409,
                          (2,2,0): 0.530,
                          (2,2,1): 0.588,
                          (3,0,0): 0.586,
                          (3,1,0): 0.642 }
           }

# Table XXII - strategy {4,3*}
GE_over_gv_realV4={ 'a094m270' : { (1,0,0): { 'mean': 0.642, 'err': 0.020 },
                                   (1,1,0): { 'mean': 0.461, 'err': 0.021 },
                                   (1,1,1): { 'mean': 0.356, 'err': 0.016 },
                                   (2,0,0): { 'mean': 0.245, 'err': 0.021 },
                                   (2,1,0): { 'mean': 0.214, 'err': 0.014 },
                                   (2,1,1): { 'mean': 0.184, 'err': 0.011 },
                                   (2,2,0): { 'mean': 0.137, 'err': 0.016 },
                                   (2,2,1): { 'mean': 0.128, 'err': 0.014 },
                                   (3,0,0): { 'mean': 0.094, 'err': 0.043 },
                                   (3,1,0): { 'mean': 0.125, 'err': 0.024 } },
                    'a094m270L': { (1,0,0): { 'mean': 0.817, 'err': 0.004 },
                                   (1,1,0): { 'mean': 0.680, 'err': 0.006 },
                                   (1,1,1): { 'mean': 0.576, 'err': 0.008 },
                                   (2,0,0): { 'mean': 0.499, 'err': 0.008 },
                                   (2,1,0): { 'mean': 0.436, 'err': 0.008 },
                                   (2,1,1): { 'mean': 0.385, 'err': 0.008 },
                                   (2,2,0): { 'mean': 0.310, 'err': 0.008 },
                                   (2,2,1): { 'mean': 0.278, 'err': 0.008 },
                                   (3,0,0): { 'mean': 0.284, 'err': 0.008 },
                                   (3,1,0): { 'mean': 0.261, 'err': 0.007 } }
                    }

# Table XXIII - strategy {4,3*}
GE_over_gv_imagVi={ 'a094m270' : { (1,0,0): { 'mean': 0.701, 'err': 0.089 },
                                   (1,1,0): { 'mean': 0.534, 'err': 0.052 },
                                   (1,1,1): { 'mean': 0.394, 'err': 0.025 },
                                   (2,0,0): { 'mean': 0.296, 'err': 0.038 },
                                   (2,1,0): { 'mean': 0.254, 'err': 0.025 },
                                   (2,1,1): { 'mean': 0.185, 'err': 0.025 },
                                   (2,2,0): { 'mean': 0.163, 'err': 0.027 },
                                   (2,2,1): { 'mean': 0.095, 'err': 0.036 },
                                   (3,0,0): { 'mean': 0.082, 'err': 0.066 },
                                   (3,1,0): { 'mean': 0.211, 'err': 0.093 } },
                    'a094m270L': { (1,0,0): { 'mean': 0.817, 'err': 0.019 },
                                   (1,1,0): { 'mean': 0.690, 'err': 0.017 },
                                   (1,1,1): { 'mean': 0.593, 'err': 0.018 },
                                   (2,0,0): { 'mean': 0.531, 'err': 0.017 },
                                   (2,1,0): { 'mean': 0.466, 'err': 0.014 },
                                   (2,1,1): { 'mean': 0.419, 'err': 0.015 },
                                   (2,2,0): { 'mean': 0.345, 'err': 0.013 },
                                   (2,2,1): { 'mean': 0.311, 'err': 0.015 },
                                   (3,0,0): { 'mean': 0.322, 'err': 0.016 },
                                   (3,1,0): { 'mean': 0.302, 'err': 0.013 } }
                    }

# Table XXIV - strategy {4,3*}
GM_over_gv_realVi={ 'a094m270' : { (1,0,0): { 'mean': 3.072, 'err': 0.058 },
                                   (1,1,0): { 'mean': 2.320, 'err': 0.071 },
                                   (1,1,1): { 'mean': 1.834, 'err': 0.068 },
                                   (2,0,0): { 'mean': 1.593, 'err': 0.069 },
                                   (2,1,0): { 'mean': 1.333, 'err': 0.048 },
                                   (2,1,1): { 'mean': 1.146, 'err': 0.051 },
                                   (2,2,0): { 'mean': 0.88, 'err': 0.015 },
                                   (2,2,1): { 'mean': 0.817, 'err': 0.067 },
                                   (3,0,0): { 'mean': 0.85, 'err': 0.020 },
                                   (3,1,0): { 'mean': 0.82, 'err': 0.014 } },
                    'a094m270L': { (1,0,0): { 'mean': 3.713, 'err': 0.027 },
                                   (1,1,0): { 'mean': 3.193, 'err': 0.022 },
                                   (1,1,1): { 'mean': 2.790, 'err': 0.023 },
                                   (2,0,0): { 'mean': 2.471, 'err': 0.024 },
                                   (2,1,0): { 'mean': 2.203, 'err': 0.027 },
                                   (2,1,1): { 'mean': 1.969, 'err': 0.031 },
                                   (2,2,0): { 'mean': 1.656, 'err': 0.029 },
                                   (2,2,1): { 'mean': 1.500, 'err': 0.035 },
                                   (3,0,0): { 'mean': 1.556, 'err': 0.028 },
                                   (3,1,0): { 'mean': 1.430, 'err': 0.029 } }
                    }

for qvec, data in GM_over_gv_realVi['a094m270'].items():
    axGM.errorbar(qvecToQ2['a094m270'][qvec],data['mean'],yerr=data['err'],\
                  color='blue',fmt='o',mfc=None,capsize=2)
    

for qvec, data in GE_over_gv_realV4['a094m270'].items():
    axGE.errorbar(qvecToQ2['a094m270'][qvec],data['mean'],yerr=data['err'],\
                  color='red',fmt='o',mfc=None,capsize=2)

for qvec, data in GE_over_gv_imagVi['a094m270'].items():
    axGE.errorbar(qvecToQ2['a094m270'][qvec]+0.02,data['mean'],yerr=data['err'],\
                  color='red',fmt='^',mfc=None,capsize=2)
# Dummy points for labels
axGM.errorbar(-1.0,0.0,yerr=0.0,color='blue',fmt='o',mfc=None,capsize=2,\
              label=r'${\tt PRD\ 105,\ 054505\ (2022)}$')
axGE.errorbar(-1.0,0.0,yerr=0.0,color='red',fmt='o',mfc=None,capsize=2,\
              label=r'${\tt PRD\ 105,\ 054505\ (2022)}$')
axGE.errorbar(-1.0,0.0,yerr=0.0,color='red',fmt='^',mfc=None,capsize=2,\
              label=r'${\tt PRD\ 105,\ 054505\ (2022)}$')

##########################################################################################
##########################################################################################



    


# From GPD analysis
#  A1 == F1
#  A4 == F1 + F2
def GE(A1,A4,Q2):
    return A1-((Q**2)/(4*mass**2))*(A4-A1)
    # return F1(Q2)-((Q**2)/(4*mass**2))*F2(Q2)
def GM(A4,Q2):
    return A4
    # return F1(Q2)+F2(Q2)






# amps={'A1': {}, 'A4': {} }
amps={'A1': [], 'A4': [] }


h=h5py.File(options.ampH5File,'r')
h2pt=h5py.File(options.twoPtH5File,'r')
mass=h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p000/'].get('tfit_4-18')[0]

for k,v in amps.items():
    # Loop over momenta in file
    for pfpi in [options.selectPfPi] if options.selectPfPi else h['/%s/mean/%s'%(k,comp)].keys():
        tmpPf=pfpi.split('_')[0][2:]
        tmpPi=pfpi.split('_')[1][2:]
        pf='%s.%s.%s'%pitd_util.as_tuple(tmpPf)
        pi='%s.%s.%s'%pitd_util.as_tuple(tmpPi)
        
        Ef=Ei=0.0;
        
        for t in h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%tmpPf].keys():
            Ef=h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%tmpPf].get(t)[0]
        for t in h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%tmpPi].keys():
            Ei=h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%tmpPi].get(t)[0]
        
        

        # Mass converted from dimensionless to GeV in momTrans call
        Q2=-1.0*pitd_util.momTrans(Ef,pf,Ei,pi,mass,32,0.094,computeFromDispReln=True)
        print(Q2)
        

        mean, err = h['/%s/mean/%s/%s/zsep000/gamma--1/tfit_4-14'%(k,comp,pfpi)]
        
        v.append( (Q2,mean,err) )


# Convert mass to GeV
mass*=(pitd_util.HBARC/0.094)
print(mass)



# Make my versions of GE & GM
amps.update({'GE': [], 'GM': []})




for n,t in enumerate(amps['A1']):
    Q2, avgA1, errA1 = t
    avgA4, errA4 = amps['A4'][n][1], amps['A4'][n][2]


    factor=Q2/(4*mass**2)
    
    avgGE=avgA1-factor*(avgA4-avgA1)
    errGE=np.sqrt( np.power(1+factor,2)*errA1**2 + np.power(-factor,2)*errA4**2 )

    amps['GE'].append( (Q2, ZV*avgGE, ZV*errGE) )
    amps['GM'].append( (Q2, ZV*avgA4, ZV*errA4) )


print(amps)
print(mass)


# Plot my values of GE/GM
for n in range(0,len(amps['GE'])):
    axGE.errorbar(amps['GE'][n][0],amps['GE'][n][1],yerr=amps['GE'][n][2],\
                  color='gray',label=r'$%s$'%options.selectPfPi if options.selectPfPi else '')
    axGM.errorbar(amps['GM'][n][0],amps['GM'][n][1],yerr=amps['GM'][n][2],\
                  color='gray',label=r'$%s$'%options.selectPfPi if options.selectPfPi else '')
    print(n)
# Dummy points for labels
axGE.errorbar(-1.0,0.0,yerr=0.0,color='gray',label='This work')
axGM.errorbar(-1.0,0.0,yerr=0.0,color='gray',label='This work')


axGE.legend()
axGM.legend()


figGE.savefig("GE-compare.pdf")
figGM.savefig("GM-compare.pdf")


    
plt.show()
