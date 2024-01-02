#!/usr/bin/python

import numpy as np

HBARC=0.1973269804
class pgitd:
    def __init__(self,snk,src,mass,a=None,L=32):
        '''
        Accepts mass passed in lattice units
        '''
        self.snk=[int(s) for s in snk.split('.')]
        self.src=[int(s) for s in src.split('.')]
        self.M=mass
        self.L=L
        self.a=a
        self.xi=None
        self.t=None

    def dispersion(self,p):
        dotP=0.0
        for n in range(0,3):
            dotP+=int(p[n])**2
        dotP*=np.power((2*np.pi/self.L),2)
        return np.sqrt(self.M**2+dotP)

    # Determine the skewness
    def skewness(self):
        try:
            self.xi=(1.0*(self.src[2]-self.snk[2]))/(self.src[2]+self.snk[2])
        except:
            self.xi=-999999

    # Determine the momentum transfer
    def transfer(self):
        Ei=self.dispersion(self.src)
        Ef=self.dispersion(self.snk)

        # Compute inner product of momenta
        dot=0.0
        for n,v in enumerate(self.snk):
            dot+=int(v)*int(self.src[n])
        dot*=np.power(2*np.pi/self.L,2)
        
        self.t=2*(self.M**2-Ei*Ef+dot)*(np.power(HBARC/self.a,2))
        
        # self.t=2*(self.mass**2-(np.sqrt(self.mass**2+self.src[0]**2+self.src[1]**2\
        #                                 +self.src[2]**2)*\
        #                         np.sqrt(self.mass**2+self.snk[0]**2+self.snk[1]**2\
        #                                 +self.snk[2]**2)-\
        #                         self.src[0]*self.snk[0]-self.src[1]*self.snk[1]-\
        #                         self.src[2]*self.snk[2]))
