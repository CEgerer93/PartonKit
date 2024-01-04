#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def logPrior(x,x0,mu,sig):
    return (1.0/((x-x0)*np.sqrt(2*np.pi)*sig))*np.exp(-1.0*((np.log(x-x0)-mu)**2)/(2*sig**2))

x0=[-1,0]
mu=[0.0,3.0]
sig=[0.4,1.0]
X=np.linspace(-1,3,1000)

cols=['red','blue']
labs=['alpha','beta']

for n,x in enumerate(x0):
    logMu=np.log(((mu[n]-x)**2)/np.sqrt((mu[n]-x)**2+sig[n]**2))
    logSig=np.sqrt(np.log(1+(sig[n]**2)/((mu[n]-x)**2)))
    plt.plot(X,logPrior(X,x,logMu,logSig),color=cols[n],label=labs[n])

# Scale up the prior widths
sig=[s*2 for s in sig]

for n,x in enumerate(x0):
    logMu=np.log(((mu[n]-x)**2)/np.sqrt((mu[n]-x)**2+sig[n]**2))
    logSig=np.sqrt(np.log(1+(sig[n]**2)/((mu[n]-x)**2)))
    plt.plot(X,logPrior(X,x,logMu,logSig),color=cols[n],label=labs[n]+'-2x',ls='--')

plt.ylim([-0.5,1.5])
plt.legend()
plt.show()
