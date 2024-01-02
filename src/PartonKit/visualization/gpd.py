#!/usr/bin/python3.8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import scipy.special as spec
import scipy.integrate as integrate
import pylab
import sys,optparse
from collections import OrderedDict

import pheno
import pdf_utils
import gpd_utils
from common_fig import *

import lhapdf
# lhapdf.pathsPrepend("/home/colin/share/LHAPDF")
lhapdf.pathsPrepend("/home/colin/LHAPDF-6.3.0/share/LHAPDF/LHAPDF")

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-A", "--ampFit", type="str", default="",
                  help='Amplitude that is fit (default = "")')
parser.add_option("-o", "--jacobiPGITDValence", type="str", default="",
                  help='Jacobi approx of pGITD; results to real jackknife samples (default = "")')
parser.add_option("-O", "--jacobiPGITDPlus", type="str", default="",
                  help='Jacobi approx of pGITD; results to imag jackknife samples (default = "")')
parser.add_option("-a", "--jacobiCorrections", type="str", default="",
                  help='Jacobi approx of pGITD corrections; <numLT>.<numAZ>.<numT4>.<numT6> (default = "")')
parser.add_option("-H", "--pGITDH5s", type="string", default="",
                  help='H5 file(s) to plot <x>:<x> (default = "x:x")')
parser.add_option("-d", "--dtypeName", type="string", default="",
                  help='Datatype name(s) to access <d>:<d> (default = "d:d")')
parser.add_option("--fit2pt",type="str", default='',
                  help='H5 file containing 2pt fits (default = '')')
parser.add_option("-z", "--zRange", type="string", default="",
                  help='Min/Max zsep in h5 <zmin>.<zmax> (default = '')')
parser.add_option("-p", "--boostMoms", type="str", default='',
                  help='Fin/Ini momenta of rpGITD numerator <pf/pi>=X.X.X/X.X.X (default = '')')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("--showPheno", action="store_true", default=False, dest="showPheno",
                  help='Show pheno PDFs (default=0)')
parser.add_option("-Q", "--phenoQ", type="int", default=1,
                  help='Evaluate pheno pdfs at <phenoQ GeV> (default = 1)')
parser.add_option("-i", "--showPDFInset", type="int", default=0,
                  help='Make PDF inset visible (default = 0)')
parser.add_option("--maxZ", type="int", default=8,
                  help="Max Z in dataset (default = 8)")
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")
parser.add_option("-r", action="store_false", default=True, dest="corrSwitch",
                  help='Fits are correlated (default = True')


# Parse the input arguments
(options, args) = parser.parse_args()

gauge_configs=options.cfgs
DIRAC=options.insertion
pdf_utils.alphaS = options.alphaS # SET ALPHAS in pdf_utils
zmin, zmax=tuple(int(z) for z in options.zRange.split('.'))
pf, pi=tuple(p for p in options.boostMoms.split('/'))




pgitdData=gpd_utils.h5Plot(options.pGITDH5s,options.dtypeName,options.fit2pt,gauge_configs,\
                           pf,pi,zmin,zmax,options.Lx)
pgitdData.initAxes()
pgitdData.read2pts()
pgitdData.showPGITDData()

print(pgitdData.ampDict)

for J in [(options.jacobiPGITDValence,'v','Re',1), (options.jacobiPGITDPlus,'+','Im',0)]: # N.B. IM SHOULD HAVE '0' instead of '1'!!!
    corrStart = J[3]
    if J[0] != "":
        for nparam, paramFile in enumerate(J[0].split('@')):

            numLT = int(options.jacobiCorrections.split('.')[0])
            numAZ = int(options.jacobiCorrections.split('.')[1])
            numT4 = int(options.jacobiCorrections.split('.')[2])
            numT6 = int(options.jacobiCorrections.split('.')[3])

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
            
            print(len(paramOrder))
            # Read fit parameters from this paramFile
            with open(paramFile) as ptr:
                for cnt, line in enumerate(ptr):
                    L=line.split(' ')
                    for n, k in enumerate(paramOrder):
                        fitParams[k].append(float(L[n]))


            # Instantiate a pitdJacobi instance, w/ pdf + corrections in real pitd
            jacColors=['purple', 'orange', 'saddlebrown','darkblue','darkgreen','black']

            
            jac=gpd_utils.pgitdJacobi(J[1],jacColors,\
                                      {k: fitParams[k] for k,v in fitParams.items()},\
                                      [numLT,numAZ,numT4,numT6],bool(options.corrSwitch),\
                                      gauge_configs,J[2],corrStart,dirac=DIRAC)
            jac.parse()
            print("For %s Jacobi..."%J[2])
            jac.printFitParams()


            if options.jacobiPGITDValence != "":
                jac.plotPGITD(pgitdData.ampDict[options.ampFit]['real']['ax'],'red')
            elif options.jacobiPGITDPlus != "":
                jac.plotPGITD(pgitdData.ampDict[options.ampFit]['imag']['ax'],'red')


            
            # # Finally plot the PDFs from jacobi polynomial fits
            # if J[2] == 'Re':
            #     if options.showPDFInset:
            #         jac.plotPDFs(ax_qv_pdf_inset,True)
            #     jac.plotPDFs(ax_pdf_qv)
            #     jac.plotXPDFs(ax_xpdf_qv)
            #     jac.plotCovHMap(fig_re_cov_heat,ax_re_cov_heat,'coolwarm') #bwr')
            #     jac.plotITDs(ax_itd_real_and_fit)
            # else:
            #     if options.showPDFInset:
            #         jac.plotPDFs(ax_qplus_pdf_inset,True)
            #     jac.plotPDFs(ax_pdf_qplus)
            #     jac.plotXPDFs(ax_xpdf_qplus)
            #     jac.plotCovHMap(fig_im_cov_heat,ax_im_cov_heat,'coolwarm') #bwr')
            #     jac.plotITDs(ax_itd_imag_and_fit)


            # perZRanges=[]
            # # Plot each jacobi fit band for a given zsep\in{1,16} atop pitd data
            # if DIRAC == 8:
            #     perZRanges=[0.95,0.85,0.65,0.4,0.2,0,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
            # if DIRAC == 11:
            #     # perZRanges=[0.94,0.75,0.55,0.3,0.2,0,-0.3,-0.3] # w/o SVD
            #     perZRanges=[0.93,0.75,0.55,0.3,0.15,-0.05,-0.3,-0.3] # w/ SVD
                
            # for z, ax in enumerate(ax_pitd_re_perz.flat):
            #     ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
            #     ax.set_ylim(perZRanges[z],1)
            #     # jac.plotPITD(ax,z+1,dispColors[z+1])
            #     jac.plotPITD(ax,z+1,\
            #                  col=dispColors[z+1] if (z+1 >= int(options.h5ZCut.split('.')[0])\
            #                                        and z+1 <= int(options.h5ZCut.split('.')[1]))\
            #                  else 'gray')

            #     pdf_utils.h5Plot(options.pITD,options.h5ZCut,options.h5PCut,[ax],\
            #                [None],gauge_configs,'pitd',dispColors,\
            #                dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy,\
            #                showZSepOnPlot=True,sysErrH5File=sysErrPITD,\
            #                plotSingleZ=z+1)

            # if DIRAC == 8:
            #     perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7,0.75,0.8,1.1,1.1]
            # if DIRAC == 11:
            #     # perZRanges=[-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
            #     perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7]
                
            # for z, ax in enumerate(ax_pitd_im_perz.flat):
            #     ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
            #     ax.set_ylim(0,perZRanges[z])
            #     jac.plotPITD(ax,z+1,dispColors[z+1])
            #     pdf_utils.h5Plot(options.pITD,options.h5ZCut,options.h5PCut,[None],\
            #                [ax],gauge_configs,'pitd',dispColors,\
            #                dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy,\
            #                showZSepOnPlot=True,sysErrH5File=sysErrPITD,\
            #                plotSingleZ=z+1)



plt.show()
