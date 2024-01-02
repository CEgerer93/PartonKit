#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pylab
import sys,optparse

import pdf_utils as PDF


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("-a", "--alpha", type="float", default=1.0,
                  help='Jacobi basis - alpha (default = 1.0)')
parser.add_option("-b", "--beta", type="float", default=1.0,
                  help='Jacobi basis - beta (default = 1.0)')
parser.add_option("-c", "--component", type="int", default=-1,
                  help='Sigma(0) -or- Eta(1) (default = -1)')
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")
# Parse the input arguments
(options, args) = parser.parse_args()

alpha=options.alpha
beta=options.beta
PDF.alphaS = options.alphaS # SET ALPHAS in pdf_utils

################
# INITIALIZE GLOBAL PROPERTIES OF FIGURES
################
# Finalize the figures
plt.rc('text', usetex=True)
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rc('xtick.major',size=5)
plt.rc('ytick.major',size=5)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('axes',labelsize=32)
truthTransparent=False
FrameAlpha=0.7
legendFaceColor="white"
# Optionally swap default black labels for white
if options.lightBkgd == 0:
    truthTransparent=True
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rc('axes',edgecolor='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('text',color='white')
    FrameAlpha=0.0
    legendFaceColor="#1b212c"

    
fig=plt.figure(figsize=(12,10)); figNLO=plt.figure(figsize=(12,10)); figJac=plt.figure(figsize=(12,10))
ax=fig.gca(); axNLO=figNLO.gca(); axJac=figJac.gca()
for a in [ax,axNLO]:
    a.set_xlim([0,16])
    a.set_xticks(np.arange(0,17,2))
    a.set_xlabel(r'$\nu$')
    a.yaxis.set_major_formatter(FormatStrFormatter(r'$%.3f$')) # Ytick formatter

axNLO.set_ylim([-0.08,0.06])
axJac.set_xlim([0,1]); axJac.set_ylim([-1,5])
axJac.set_xlabel(r'$x$'); axJac.set_ylabel(r'$x^\alpha\left(1-x\right)^\beta\Omega_n^{\left(\alpha,\beta\right)}\left(x\right)$')

if options.component == 0:
    ax.set_ylabel(r'$\sigma_{0,n}^{\left(\alpha,\beta\right)}\left(\nu\right)$')
    axNLO.set_ylabel(r'$\sigma_{n,{\rm NLO}}^{\left(\alpha,\beta\right)}\left(\nu,z^2\mu^2\right)$')
if options.component == 1:
    ax.set_ylabel(r'$\eta_{0,n}^{\left(\alpha,\beta\right)}\left(\nu\right)$')
    axNLO.set_ylabel(r'$\eta_{n,{\rm NLO}}^{\left(\alpha,\beta\right)}\left(\nu,z^2\mu^2\right)$')

nu=np.linspace(0,20,1000); x=np.linspace(0,1,500)
for n in range(0,8):
    jacobiPDF=[np.power(xi,alpha)*np.power(1-xi,beta)*PDF.Jacobi(n,alpha,beta,xi) for xi in x]
    
    poly=None; polyNLO=None
    if options.component == 0:
        poly=PDF.pitd_texp_sigma_n_treelevel(n,75,alpha,beta,nu)
        polyNLO=PDF.pitd_texp_sigma_n(n,75,alpha,beta,nu,2)-PDF.pitd_texp_sigma_n_treelevel(n,75,alpha,beta,nu)
    if options.component == 1:
        poly=PDF.pitd_texp_eta_n_treelevel(n,75,alpha,beta,nu)
        polyNLO=PDF.pitd_texp_eta_n(n,75,alpha,beta,nu,2)-PDF.pitd_texp_eta_n_treelevel(n,75,alpha,beta,nu)
        
    ax.plot(nu,poly,label=r'$n=%d$'%n)
    axNLO.plot(nu,polyNLO,label=r'$n=%d$'%n)
    axJac.plot(x,jacobiPDF,label=r'$n=%d$'%n)

ax.legend(loc='upper right',ncol=2,columnspacing=0.5,handletextpad=0.2,framealpha=FrameAlpha)
axNLO.legend(loc='lower left',ncol=2,columnspacing=0.5,handletextpad=0.2,framealpha=FrameAlpha)
axJac.legend(loc='upper right',ncol=2,columnspacing=0.5,handletextpad=0.2,framealpha=FrameAlpha)


figName=''
if options.component == 0:
    figName='sigma_0n'
if options.component == 1:
    figName='eta_0n'


suffix=''
form='pdf'
if options.lightBkgd == 0:
    suffix='.dark'
    form='png'
    
    
fig.savefig("%s%s.%s"%(figName,suffix,form),dpi=500,transparent=truthTransparent,\
            bbox_inches='tight',pad_inches=0.1,format=form)
figNLO.savefig("%s_NLO%s.%s"%(figName,suffix,form),dpi=500,transparent=truthTransparent,\
            bbox_inches='tight',pad_inches=0.1,format=form)
figJac.savefig("someJac%s.%s"%(suffix,form),dpi=500,transparent=truthTransparent,\
               bbox_inches='tight',pad_inches=0.1,format=form)
plt.show()
