#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys,optparse
sys.path.append("/home/colin/QCD/pseudoDists/strFuncViz")
sys.path.append("/home/colin/QCD/pseudoDists")
import pdf_utils
import common_fig
from scipy import integrate
from scipy import special as spec

pdf_utils.alphaS=0.303

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);
# parser.add_option("-z", type="int", default=-1,
#                   help='Z-sep to consider (default = -1)')
parser.add_option("--xi", type="float", default=0.0,
                  help='Skewness (default = 0.0)')
parser.add_option("-a", "--alpha", type="float", default=-0.5,
                  help='Alpha of x^a(1-x)^b (default = -0.5)')
parser.add_option("-b", "--beta", type="float", default=2.0,
                  help='Beta of x^a(1-x)^b (default = 2.0)')
# Parse the input arguments
(options, args) = parser.parse_args()



print("MU = %.5f"%pdf_utils.MU)



# NU=np.linspace(0,3,100)
NU=np.linspace(0,20,100)


def gpd(x):
    return np.power(x,options.alpha)*np.power(1-x,options.beta)*(1.0/spec.beta(options.alpha+1,options.beta+1))



fig2 = plt.figure(figsize=(10,6)); ax2=fig2.gca()
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$H(x,\xi,t,\mu^2)$')
ax2.set_ylim([-0.1,4])

# X=np.linspace(0,1,300)
X=np.linspace(0,1,10)
GPD=[gpd(x) for x in X]
ax2.plot(X,GPD)


###### GITD Stuff
fig3 = plt.figure(figsize=(10,6)); ax3=fig3.gca()
ax3.set_xlabel(r'$\nu$')
ax3.set_ylabel(r'$\widetilde{\mathcal{I}}(\nu,\xi,t,\mu^2)$')
def gitd(nu):
    sr = lambda x: np.cos(nu*x)*gpd(x)
    si = lambda x: np.sin(nu*x)*gpd(x)
    # return integrate.quad(sr,0,1)[0]
    return integrate.quad(si,0,1)[0]
    # return complex(integrate.quad(sr,0,1)[0],integrate.quad(si,0,1)[0])
GITD=[gitd(nu) for nu in NU]
ax3.plot(NU,GITD)


########### Convolution stuff
fig4 = plt.figure(figsize=(10,6)); ax4=fig4.gca()
ax4.set_xlabel(r'$\nu$')
ax4.set_ylabel(r'$K\otimes\widetilde{\mathcal{I}}$')
ax4.set_xlim([-0.05,21])



def BconvolGITD(n,nu,xi):
    ss = lambda u: (2*u/(1-u))*np.cos((1-u)*xi*nu)*(gitd(u*nu)-GITD[n])+(np.sin((1-u)*xi*nu)/(xi*nu))*gitd(u*nu) if xi != 0.0\
        else (2*u/(1-u))*np.cos((1-u)*xi*nu)*(gitd(u*nu)-GITD[n])+(1-u)*gitd(u*nu)
    # \delta(1-u) within scale-dependent matching kernel returns -GITD/2 at \nu
    return integrate.quad(ss,0,1)[0] - 0.5*GITD[n]

def LconvolGITD(n,nu,xi):
    ss = lambda u: (4.0*np.log(1-u)/(1-u))*np.cos((1-u)*xi*nu)*(gitd(u*nu)-GITD[n])-(2.0*np.sin((1-u)*xi*nu)/(xi*nu))*gitd(u*nu) if xi != 0.0\
        else (4.0*np.log(1-u)/(1-u))*np.cos((1-u)*xi*nu)*(gitd(u*nu)-GITD[n])-2.0*(1-u)*gitd(u*nu)
    # \delta(1-u) within scale-independent matching kernel returns GITD at \nu
    return integrate.quad(ss,0,1)[0] + GITD[n]


################################################
# Plot for multiple values of skewness
################################################
styles=['solid','dotted','dashed','dashdot']
for n,xi in enumerate([0,1.0/3,0.5,1]):
    BOTimesGITD=[BconvolGITD(n,nu,xi) for n,nu in enumerate(NU)]
    LOTimesGITD=[LconvolGITD(n,nu,xi) for n,nu in enumerate(NU)]

    labelB=r'$B\otimes\widetilde{\mathcal{I}}$' if xi==0 else ''
    labelL=r'$L\otimes\widetilde{\mathcal{I}}$' if xi==0 else ''
    
    ax4.plot(NU,BOTimesGITD,color='orange',label=labelB,ls=styles[n])
    ax4.plot(NU,LOTimesGITD,color='black',label=labelL,ls=styles[n])

    #################
    #################
    # REAL
    #################
    #################
    # # Default alpha/beta
    # ax4.plot([10,12],[-2-n*0.15,-2-n*0.15],color='gray',ls=styles[n])
    # ax4.text(12.2,-2-n*0.15,r'$\xi=%.3f$'%xi,fontsize=16,verticalalignment='center')
    # # # Default alpha=0.5 / beta=4.0
    # # ax4.plot([10,12],[-3.25-n*0.25,-3.25-n*0.25],color='gray',ls=styles[n])
    # # ax4.text(12.2,-3.25-n*0.25,r'$\xi=%.3f$'%xi,fontsize=16,verticalalignment='center')
    #################
    #################
    # IMAG
    #################
    #################
    # # Default alpha/beta
    # ax4.plot([10,12],[1.05-n*0.1,1.05-n*0.1],color='gray',ls=styles[n])
    # ax4.text(12.2,1.05-n*0.1,r'$\xi=%.3f$'%xi,fontsize=16,verticalalignment='center')
    # # Default alpha=0.5 / beta=4.0
    ax4.plot([10,12],[1.8-n*0.2,1.8-n*0.2],color='gray',ls=styles[n])
    ax4.text(12.2,1.8-n*0.2,r'$\xi=%.3f$'%xi,fontsize=16,verticalalignment='center')
    
ax4.legend()




# Determine resulting rpGITD
fig5 = plt.figure(figsize=(12,8)); ax5=fig5.gca()
ax5.set_xlabel(r'$\nu$')
ax5.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{M}\left(\nu,\xi,t,z^2\right)$')
for z in range(1,9):
    RPGITD=[i - ((pdf_utils.alphaS*pdf_utils.Cf)/(2*np.pi))*(np.log( (np.exp(2*np.euler_gamma+1)/4)*np.power(pdf_utils.MU*z,2) )*BOTimesGITD[n]+LOTimesGITD[n]) for n,i in enumerate(GITD) ]
    ax5.plot(NU,RPGITD,label=r'$z=%i$'%z)
ax5.legend()


# totITDCorrection=[((-pdf_utils.alphaS*pdf_utils.Cf)/(2*np.pi))\
#                   *((np.log(np.power(options.z*pdf_utils.muRenorm,2)/4)+1+2*np.euler_gamma)\
#                     *b+l) for b,l in zip(BOTimesITD,LOTimesITD)]

# ############### Ratio of convolution to orig ITD
# fig5 = plt.figure(figsize=(10,6)); ax5=fig5.gca()
# ax5.set_xlabel(r'$\nu$')
# ax5.set_ylabel(r'$Corr$')

# # ax5.plot(NU,[a/b for a,b in zip(BOTimesITD,ITD)],color='orange')
# ax5.plot(NU,totITDCorrection,color='black',label='Correction to ITD')
# ax5.plot(NU,[n/d for n,d in zip(totITDCorrection,ITD)],color='gray',label='Corr/ITD Ratio')
# ax5.legend(loc='upper right')


fig4.savefig("KCrossI_xi%.3f_alpha%.3f_beta%.3f.pdf"%(options.xi,options.alpha,options.beta),bbox_inches='tight',pad_inches=0.1)
fig5.savefig("rpGITD-xi%.3f_from_modeled-GPD-alpha%.3f_beta%.3f.pdf"%(options.xi,options.alpha,options.beta),bbox_inches='tight',pad_inches=0.1)

plt.show()






