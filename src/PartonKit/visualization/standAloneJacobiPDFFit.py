#!/usr/bin/python3

#######################
# Compare a pdf parameterization (A) to some other pdf parameterization (B)
#
# Comparison is a normalization of (A) by central value of (B)
#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdf_utils as PDF
import parse_util
import pylab
import sys,optparse

from common_fig import *

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-j", "--jacobiFitResFile", type="str", default="",
                  help='Jacobi fit results (default = '')')
parser.add_option("-a", "--jacobiCorrections", type="str", default="",
                  help='Jacobi approx of pITD corrections; <numLT>.<numAZ>.<numT4>.<numT6>.<numT8>.<numT10> (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-t", "--pdfType", type="str", default="",
                  help='PDF type < v -or- + > (default='')')
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")
parser.add_option("-c", "--cfgs", type="int", default=349,
                  help='Gauge configs (default = "")')
parser.add_option("-p", "--pITD", type="str", default="",
                  help='Pseudo-ITD  -  raw lattice data H5 (default = "")')
parser.add_option("--sysErrPITD", type="str", default="",
                  help='H5 for systematic error inclusion (default = "")')
parser.add_option("--oldH5Hierarchy", action="store_true", default=False, dest="h5Hierarchy",
                  help='Read associated h5s in old format (b4 zsep/pf_pi changes)')

# Parse the input arguments
(options, args) = parser.parse_args()

labelFontSize=14
DIRAC=options.insertion
paramFile=options.jacobiFitResFile
PDF.alphaS = options.alphaS # SET ALPHAS in pdf_utils

# Color according the z value
mainColors=[ cm.gist_rainbow(d) for d in np.linspace(0,1,9) ]

    
###############
# INSTANTIATE FIGURES
###############
fig=plt.figure(figsize=(12,10))
ax=fig.gca()

fig_pitd_perz, ax_pitd_perz=plt.subplots(2,4,figsize=(18,14))
ax_pitd_perz_all = fig_pitd_perz.gca()
ax_pitd_perz_all.tick_params(labelcolor='none', top=False, bottom=False,\
                             left=False, right=False)
ax_pitd_perz_all.set_xlabel(r'$\nu$')



def jacobiParamContainers(corrections,corrStart):
    
    numLT = int(corrections.split('.')[0])
    numAZ = int(corrections.split('.')[1])
    numT4 = int(corrections.split('.')[2])
    numT6 = int(corrections.split('.')[3])
    numT8 = int(options.jacobiCorrections.split('.')[4])
    numT10 = int(options.jacobiCorrections.split('.')[5])
    
    # Reset/form the fitParams dict & paramOrder list
    fitParams={'L2': [], 'chi2': [], 'L2/dof': [], 'chi2/dof': [], '\\alpha': [], '\\beta': []}
    paramOrder=['L2', 'chi2', 'L2/dof', 'chi2/dof', '\\alpha', '\\beta']
    for l in range(0,numLT):
        fitParams.update({'C^{lt}_%d'%l: []})
        paramOrder.append('C^{lt}_%d'%l)
    for a in range(corrStart,numAZ+corrStart):
        fitParams.update({'C^{az}_%d'%a: []})
        paramOrder.append('C^{az}_%d'%a)
    for t in range(corrStart,numT4+corrStart):
        fitParams.update({'C^{t4}_%d'%t: []})
        paramOrder.append('C^{t4}_%d'%t)
    for s in range(corrStart,numT6+corrStart):
        fitParams.update({'C^{t6}_%d'%s: []})
        paramOrder.append('C^{t6}_%d'%s)
    for u in range(corrStart,numT8+corrStart):
        fitParams.update({'C^{t8}_%d'%u: []})
        paramOrder.append('C^{t8}_%d'%u)
    for v in range(corrStart,numT10+corrStart):
        fitParams.update({'C^{t10}_%d'%v: []})
        paramOrder.append('C^{t10}_%d'%v)

    return fitParams, paramOrder


corrStart = 1
comp = 'Re'
if options.pdfType == '+':
    corrStart = 0
    comp = 'Im'


# Set perZRanges based on which pdfType and current
perZRanges=[]
if options.pdfType == 'v':
    if DIRAC == 8:
        perZRanges=[0.95,0.85,0.65,0.4,0.2,0,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
    if DIRAC == 11:
        perZRanges=[0.94,0.75,0.55,0.3,0.2,0,-0.3,-0.3]
else:
    if DIRAC == 8:
        perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7,0.75,0.8,1.1,1.1]
    if DIRAC == 11:
        perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7]
    
fitParams, order = jacobiParamContainers(options.jacobiCorrections,corrStart)

with open(paramFile) as ptr:
    for cnt, line in enumerate(ptr):
        L=line.split(' ')
        for n, k in enumerate(order):
            fitParams[k].append(float(L[n]))

# Instantiate a pitdJacobi instance
jacColors=['purple', 'orange', 'saddlebrown','darkblue','darkgreen','black']
jac=PDF.pitdJacobi(options.pdfType,jacColors,\
                   {k: fitParams[k] for k,v in fitParams.items()},\
                   [int(n) for n in options.jacobiCorrections.split('.')],\
                   True,options.cfgs,comp,corrStart,dirac=DIRAC)

jac.parse()
jac.printFitParams()

for z, ax in enumerate(ax_pitd_perz.flat):
    ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*6)
    ax.set_ylim(perZRanges[z],1) if options.pdfType == 'v' else ax.set_ylim(0,perZRanges[z])

    if options.cfgs == 1:
        jac.plotPITDNoErr(ax,z+1,mainColors[z+1])
    else:
        jac.plotPITD(ax,z+1,mainColors[z+1])

    if options.pITD != "" and options.sysErrPITD != "":
        rAx=[None]; iAx=[None]
        if options.pdfType == 'v': rAx=[ax]
        if options.pdfType == '+': iAx=[ax]
        PDF.h5Plot(options.pITD,"1.8","1.6",rAx,\
                   iAx,options.cfgs,'pitd',mainColors,\
                   dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy,\
                   showZSepOnPlot=True,sysErrH5File=options.sysErrPITD,\
                   plotSingleZ=z+1)
        

plt.show()
