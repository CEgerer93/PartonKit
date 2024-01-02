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
import pdf_utils as PDF
from common_fig import *

import lhapdf
# lhapdf.pathsPrepend("/home/colin/share/LHAPDF")
lhapdf.pathsPrepend("/home/colin/LHAPDF-6.3.0/share/LHAPDF/LHAPDF")

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-e", "--fit2ParamsValence", type="str", default="",
                  help='2-param PDF fit results to real jackknife samples (default = "")')
parser.add_option("-E", "--fit2ParamsPlus", type="str", default="",
                  help='2-param PDF fit results to imag jackknife samples (default = "")')
parser.add_option("-f", "--fit3ParamsValence", type="str", default="",
                  help='3-param PDF fit results to real jackknife samples (default = "")')
parser.add_option("-F", "--fit3ParamsPlus", type="str", default="",
                  help='3-param PDF fit results to imag jackknife samples (default = "")')
parser.add_option("-g", "--fit4ParamsValence", type="str", default="",
                  help='4-param PDF fit results to real jackknife samples (default = "")')
parser.add_option("-G", "--fit4ParamsPlus", type="str", default="",
                  help='4-param PDF fit results to imag jackknife samples (default = "")')
parser.add_option("-o", "--jacobiPITDValence", type="str", default="",
                  help='Jacobi approx of pITD; results to real jackknife samples (default = "")')
parser.add_option("-a", "--jacobiCorrections", type="str", default="",
                  help='Jacobi approx of pITD corrections; <numLT>.<numAZ>.<numT4>.<numT6>.<numT8>.<numT10> (default = "")')
parser.add_option("-O", "--jacobiPITDPlus", type="str", default="",
                  help='Jacobi approx of pITD; results to imag jackknife samples (default = "")')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-S", "--msbarITD", type="str", default="",
                  help='Evolved/Matched pITD data H5 (default = "")')
parser.add_option("-d", "--dglap", type="str", default="",
                  help='Plot effect of DGLAP kernel on pITD data <not supported atm> (default = '')')
parser.add_option("-m", "--matching", type="str", default="",
                  help='Plot effect of MATCHING kernel on pITD data <not supported atm> (default = '')')
parser.add_option("-p", "--pITD", type="str", default="",
                  help='Pseudo-ITD  -  raw lattice data H5 (default = "")')
parser.add_option("--sysErrPITD", type="str", default="",
                  help='H5 for systematic error inclusion (default = "")')
parser.add_option("-Z", "--h5ZCut", type="str", default="x.x",
                  help='Cut on <zmin>.<zmax> in any h5 file (default = x.x)')
parser.add_option("-P", "--h5PCut", type="str", default="x.x",
                  help='Cut on <pmin>.<pmax> in any h5 file (default = x.x)')
# parser.add_option("-v", "--paramCorrelations", type="int", default=0,
#                   help='Plot parameter correlations (default = 0)')
# parser.add_option("-s", "--ReImFit", type="int", default=0,
#                   help='Passed fit params are for simultaneous RE/IM (default = 0)')
parser.add_option("-k", "--kernels2", type="str", default='',
                  help='Kernels used in fitting : K (pITD - PDF) -or- C (ITD - PDF) (default = "x.x")')
parser.add_option("-K", "--kernels3", type="str", default='',
                  help='Kernels used in fitting : K (pITD - PDF) -or- C (ITD - PDF) (default = "x.x")')
parser.add_option("-L", "--kernels4", type="str", default='',
                  help='Kernels used in fitting : K (pITD - PDF) -or- C (ITD - PDF) (default = "x.x")')
parser.add_option("-r", "--corrSwitch", type="int", default=1,
                  help='Fits are correlated (default = 1)')
parser.add_option("--oldH5Hierarchy", action="store_true", default=False, dest="h5Hierarchy",
                  help='Read associated h5s in old format (b4 zsep/pf_pi changes)')
parser.add_option("-H", "--showPheno", type="int", default=0,
                  help='Show pheno PDFs (default=0)')
parser.add_option("-N", "--showPhenoITD", type="int", default=0,
                  help='Show pheno ITDs (default=0)')
parser.add_option("-J", "--computePhenoITDs", type="int", default=0,
                  help='Compute pheno ITDs (default=0)')
parser.add_option("-Q", "--phenoQ", type="int", default=1,
                  help='Evaluate pheno pdfs at <phenoQ GeV> (default = 1)')
parser.add_option("-i", "--showPDFInset", type="int", default=0,
                  help='Make PDF inset visible (default = 0)')
parser.add_option("--darkBkgd", action="store_true", default=False, dest="darkBkgd",
                  help='Format figs for dark background (default = False)')
parser.add_option("-t", "--pdfTitle", type="int", default=0,
                  help='Include pdf functional at top of figures (default = 0)')
parser.add_option("-w", "--writePhenoPDFs", type="int", default=0,
                  help='Write extracted Pheno PDFs to txt files (default = 0)')
parser.add_option("-x", "--writeITDsFromFits", type="int", default=0,
                  help='Write ITDs resulting from fitting F.T. fits of PDFs to evolved ITD data (default = 0)')
parser.add_option("--maxZ", type="int", default=8,
                  help="Max Z in dataset (default = 8)")
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")

# Parse the input arguments
(options, args) = parser.parse_args()

# To shift parameter text boxes around figures
buff=1
gauge_configs=options.cfgs
ITD=options.msbarITD
pITD=options.pITD
sysErrPITD=None if options.sysErrPITD == "" else options.sysErrPITD
DIRAC=options.insertion
PDF.alphaS = options.alphaS # SET ALPHAS in pdf_utils

AMP='M' if DIRAC == 8 else 'Y'


if options.showPheno:
    phenoPDFs=pheno.pheno(options.phenoQ,DIRAC,options.darkBkgd)
    phenoPDFs.accessPDFs()
    phenoPDFs.extractPDFs()



###########
# Instantiate figures
###########
fig_pdf_qv=plt.figure(figsize=(12,10))
fig_xpdf_qv=plt.figure(figsize=(12,10))
fig_pdf_qplus=plt.figure(figsize=(12,10))
fig_xpdf_qplus=plt.figure(figsize=(12,10))
fig_pdf_q=plt.figure(figsize=(12,10))
fig_pdf_qbar=plt.figure(figsize=(12,10))
ax_pdf_qv=fig_pdf_qv.gca()
ax_xpdf_qv=fig_xpdf_qv.gca()
ax_pdf_qplus=fig_pdf_qplus.gca()
ax_xpdf_qplus=fig_xpdf_qplus.gca()
ax_pdf_q=fig_pdf_q.gca()
ax_pdf_qbar=fig_pdf_qbar.gca()
# Set ranges of figures
ax_pdf_qv.set_xlim([0.0,1.0])
ax_pdf_qv.set_ylim([-0.5,6])
ax_pdf_qplus.set_xlim([0.0,1.0])
ax_pdf_qplus.set_ylim([-0.5,6])
ax_pdf_q.set_xlim([0.0,1.0])
ax_pdf_q.set_ylim([-0.5,6])
ax_pdf_qbar.set_xlim([0.0,1.0])
ax_pdf_qbar.set_ylim([-0.5,6])
ax_xpdf_qv.set_xlim([0.0,1.0])
ax_xpdf_qv.set_ylim([-0.2,0.6])
ax_xpdf_qplus.set_xlim([0.0,1.0])
ax_xpdf_qplus.set_ylim([0,0.6])

# Add an inset figure to fig_pdf to show PDF fits/pheno PDFs at large-x
# ax_pdf_inset=None #fig_pdf_qv.add_axes([0.25,3,0.3,0.3])
if options.showPDFInset: # optionally bring the inset into view
    ax_qv_pdf_inset=fig_pdf_qv.add_axes([0.28,0.55,0.35,0.3]) # percentages of original ax_pdf object
    ax_qplus_pdf_inset=fig_pdf_qplus.add_axes([0.28,0.55,0.35,0.3])
    for a in [ax_qv_pdf_inset, ax_qplus_pdf_inset]:
        a.set_xlim([0.51,1.0])
        a.set_ylim([-0.1,0.3])
        # a.set_yscale("symlog")

# Color according the z value
mainColors=None
# mainColors=['blue','green','red','white','purple','orange','magenta',(0.1,1,0.1),'black','gray','gray']
# mainColors=['blue','green','red','purple','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']
if not options.darkBkgd:
    mainColors=['red','purple','blue','green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']
else:
    mainColors=['red','purple',cm.twilight(40),'green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']

# dispColors = ['gray']
# for dc in [ cm.brg(d) for d in np.linspace(0,1,17) ]:
# for dc in [ cm.hsv(d) for d in np.linspace(0,1,17) ]:
# for dc in [ cm.rainbow(d) for d in np.linspace(0,1,17) ]:
# for dc in [ cm.terrain(d) for d in np.linspace(0,1,17) ]:
# for dc in [ cm.cool(d) for d in np.linspace(0,1,int(options.h5ZCut.split('.')[1])+1) ]:




######
# SPECTRAL_R WAS FOR DEFENSE COLORS
######
# for dc in [ cm.Spectral_r(d) for d in np.linspace(0,1,int(options.h5ZCut.split('.')[1])+1) ]:

dispColors = [ cm.rainbow(d) for d in np.linspace(0,1,options.maxZ+1) ]




####################################################
# ADD PHENOMENOLOGICAL PDFS TO Q/QBAR & Qv/Q+ PLOTS
####################################################
if options.showPheno:
    for phN, phK in enumerate(phenoPDFs.F):

        # # Just some hard neglect of MSTW
        # if phK == setMSTWNNLO:
        #     continue

        axesToIncludePhenos=[(ax_pdf_q,'q'), (ax_pdf_qbar,'qbar'),\
                             (ax_pdf_qv,'qv'), (ax_pdf_qplus,'q+')]
        if options.showPDFInset:
            axesToIncludePhenos.append((ax_qv_pdf_inset,'qv'))
            axesToIncludePhenos.append((ax_qplus_pdf_inset,'q+'))
        
        for axKeys, dist in axesToIncludePhenos:
            axKeys.plot(phenoPDFs.phenoX,phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                        color=phenoPDFs.F[phK]['color'],label=phenoPDFs.F[phK]['label'])
            axKeys.fill_between(phenoPDFs.phenoX,\
                                [a+e for a, e in zip(phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                                                     phenoPDFs.F[phK]['pdf2plot'][dist]['errplus'])],\
                                [a-e for a, e in zip(phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                                                     phenoPDFs.F[phK]['pdf2plot'][dist]['errminus'])],\
                                color=phenoPDFs.F[phK]['color'],alpha=phenoPDFs.phenoAlpha,\
                                label=phenoPDFs.F[phK]['label'],hatch=phenoPDFs.phenoHatch,lw=0)
    



#######################################
# ADD PHENOMENOLOGICAL PDFS TO ITD PLOT
#######################################
if options.showPhenoITD:
    if options.computePhenoITDs:
        for phN, phK in enumerate(F_ITD):
            for ax, dist in [(ax_itd_real_and_fit,'qv'), (ax_itd_imag_and_fit,'q+')]:
                ax.plot(phenoNU,F_ITD[phK]['itd2plot'][dist]['central'],color=F_ITD[phK]['color'],label=F_ITD[phK]['label'])
                ax.fill_between(phenoNU,[a+e for a, e in\
                                         zip(F_ITD[phK]['itd2plot'][dist]['central'],\
                                             F_ITD[phK]['itd2plot'][dist]['errplus'])],
                                [a-e for a, e in\
                                 zip(F_ITD[phK]['itd2plot'][dist]['central'],\
                                     F_ITD[phK]['itd2plot'][dist]['errminus'])],\
                                color=F_ITD[phK]['color'],alpha=phenoAlpha,\
                                label=F_ITD[phK]['label'],hatch=phenoHatch,lw=0)
    else:
        for ax, dist in [(ax_itd_real_and_fit,'qv'), (ax_itd_imag_and_fit,'q+')]:
            CJ15_ITD=np.loadtxt("PHENO_ITDs_derived/cj15nlo-ITD.%s.dat"%dist,delimiter=' ')
            MSTW_ITD=np.loadtxt("PHENO_ITDs_derived/mstwnnlo-ITD.%s.dat"%dist, delimiter=' ')
            JAM20_ITD=np.loadtxt("PHENO_ITDs_derived/jam20nlo-ITD.%s.dat"%dist,delimiter=' ')
            NNPDF_ITD=np.loadtxt("PHENO_ITDs_derived/nnpdfnnlo-ITD.%s.dat"%dist,delimiter=' ')
            # phenoITD=[(NNPDF_ITD,F[setNNPDFNNLO]['color'],F[setNNPDFNNLO]['label']),\
            #           (MSTW_ITD,F[setMSTWNNLO]['color'],F[setMSTWNNLO]['label']),\
            #           (CJ15_ITD,F[setCJ15NLO]['color'],F[setCJ15NLO]['label'])]
            phenoITD=[(CJ15_ITD,F[setCJ15NLO]['color'],F[setCJ15NLO]['label'])]
            
            # Grab arrays [:370] to hide weird dip in CJ derived ITD (05/27/2021)
            for n, p in enumerate(phenoITD):
                ax.plot(phenoNU,p[0][:,0],color=p[1],label=p[2])
                ax.fill_between(phenoNU,p[0][:,0]+p[0][:,1],p[0][:,0]-p[0][:,2],\
                                color=p[1],alpha=phenoAlpha,label=p[2],hatch=phenoHatch,lw=0)





# Open the raw reduced pITD data and plot contents
if options.pITD != "":
    fig_pitd_real=plt.figure(figsize=(12,10))
    ax_pitd_real=fig_pitd_real.gca()
    ax_pitd_real.set_xlim([0,16])
    ax_pitd_real.set_ylim([-0.3,1.2])
    ax_pitd_real.xaxis.set_ticks(np.arange(0,18,2))
    ax_pitd_real.set_xlabel(r'$\nu$')
    ax_pitd_real.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{%s}(\nu,z^2)$'%AMP)

    fig_pitd_re_perz, ax_pitd_re_perz=plt.subplots(2,4,figsize=(18,14)) # 2->3 for 350MeV
    fig_pitd_im_perz, ax_pitd_im_perz=plt.subplots(2,4,figsize=(18,14)) # 2->3 for 350MeV
    fig_pitd_re_perz.subplots_adjust(top=0.965,bottom=0.140,left=0.104,right=0.967,hspace=0.183,wspace=0.337)
    fig_pitd_im_perz.subplots_adjust(top=0.972,bottom=0.140,left=0.102,right=0.982,hspace=0.244,wspace=0.249)
    fig_pitd_re_perz.add_subplot(111, frameon=False)
    fig_pitd_im_perz.add_subplot(111, frameon=False)
    ax_pitd_re_perz_all=fig_pitd_re_perz.gca()
    ax_pitd_im_perz_all=fig_pitd_im_perz.gca()
    for a in [ax_pitd_re_perz_all, ax_pitd_im_perz_all]:
        a.tick_params(labelcolor='none', top=False, bottom=False,\
                      left=False, right=False)
        a.set_xlabel(r'$\nu$')
    ax_pitd_re_perz_all.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{%s}(\nu,z^2)$'%AMP,labelpad=18)
    ax_pitd_im_perz_all.set_ylabel(r'$\mathfrak{Im}\ \mathfrak{%s}(\nu,z^2)$'%AMP,labelpad=18)

    
    fig_pitd_imag=plt.figure(figsize=(12,10))
    ax_pitd_imag=fig_pitd_imag.gca()
    ax_pitd_imag.set_xlim([0,16])
    ax_pitd_imag.set_ylim([-0.3,1.2])
    ax_pitd_imag.xaxis.set_ticks(np.arange(0,18,2))
    ax_pitd_imag.set_xlabel(r'$\nu$')
    ax_pitd_imag.set_ylabel(r'$\mathfrak{Im}\ \mathfrak{%s}(\nu,z^2)$'%AMP)

    PDF.h5Plot(pITD,options.h5ZCut,options.h5PCut,[ax_pitd_real],\
               [ax_pitd_imag],options.cfgs,"pitd",dispColors,dirac=DIRAC,\
               oldH5Hierarchy=options.h5Hierarchy,sysErrH5File=sysErrPITD)


    # Add a jacobi fit param covariance heatmap
    fig_re_cov_heat = plt.figure(figsize=(12,10)); ax_re_cov_heat = fig_re_cov_heat.gca()
    fig_im_cov_heat = plt.figure(figsize=(12,10)); ax_im_cov_heat = fig_im_cov_heat.gca()

    
            
# Open the matched ITD data and plot contents
if options.msbarITD != "":
    fig_itd_real_and_fit=plt.figure(figsize=(12,10))
    fig_itd_imag_and_fit=plt.figure(figsize=(12,10))
    ax_itd_real_and_fit=fig_itd_real_and_fit.gca()
    ax_itd_imag_and_fit=fig_itd_imag_and_fit.gca()
    fig_itd_real=plt.figure(figsize=(12,10))
    fig_itd_imag=plt.figure(figsize=(12,10))
    ax_itd_real=fig_itd_real.gca()
    ax_itd_imag=fig_itd_imag.gca()

    for r in [ax_itd_real, ax_itd_real_and_fit]:
        r.set_xlim([0,16])
        r.set_ylim([-0.3,1.2])
        r.xaxis.set_ticks(np.arange(0,18,2))
        # # COMMENT BELOW - FOR ZOOMED IN LOOK AT SMALL NU BEHAVIOR
        # r.set_xlim([0,2.65])
        # r.set_ylim([0.798,1.005])
        # # COMMENT ABOVE
        r.set_xlabel(r'$\nu$')
        r.set_ylabel(r'$\mathfrak{Re}\ Q(\nu,\mu^2)$')
    for i in [ax_itd_imag, ax_itd_imag_and_fit]:
        i.set_xlim([0,16])
        i.set_ylim([-0.3,1.2])
        i.xaxis.set_ticks(np.arange(0,18,2))
        i.set_xlabel(r'$\nu$')
        i.set_ylabel(r'$\mathfrak{Im}\ Q(\nu,\mu^2)$')

    PDF.h5Plot(ITD,options.h5ZCut,options.h5PCut,[ax_itd_real,ax_itd_real_and_fit],\
               [ax_itd_imag,ax_itd_imag_and_fit],options.cfgs,"itd",\
               dispColors,dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy)




####################################################################
####################################################################
# PROCEED IF WE HAVE A 2-PARAMETER REAL FIT
####################################################################
####################################################################
if options.fit2ParamsValence != "":
    # Loop over all 2-type real fit results
    for nparam, param2File in enumerate(options.fit2ParamsValence.split('@')):

        # Reset the fit parameter dictionaries
        fitParamsAll2V={'chi2': [], 'alpha': [], 'beta': []}
        params2=['chi2', 'alpha', 'beta']

        # Read fit parameters from this param2File
        with open(param2File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                # As of 02-23-2021 Fit params include norm, which isnt needed in qval, so remove it
                del L[1]
                for n, k in enumerate(params2):
                    fitParamsAll2V[k].append(float(L[n]))

        
                
        # Get the chi2/dof
        dumc=0.0
        meanc=np.mean(fitParamsAll2V['chi2'])
        for g in range(0,gauge_configs):
            dumc+=np.power(fitParamsAll2V['chi2'][g]-meanc,2)
        dumc*=((1.0*(gauge_configs-1))/gauge_configs)

        localColors=['red','indigo']
        # Instantiate a 2-parameter valence pdf
        qv2=PDF.pdf2('v',options.kernels2.split('.')[nparam],localColors[nparam],\
                     {'chi2': fitParamsAll2V['chi2'], 'alpha': fitParamsAll2V['alpha'],\
                      'beta': fitParamsAll2V['beta'],'norm': None},bool(options.corrSwitch),\
                     gauge_configs)
        qv2.parse()
        qv2.quarkSum() # check normalization
        # Dump fit param covariances determined from parsing of jackknife data
        print("2-VALENCE Covariance")
        # print "2-VALENCE Covariance"
        qv2.printFitCov()
        qv2.printNorm()

        # print "\nBest fit parameters of REAL 2-parameter kernel = %s fit .... "\
        #     %options.kernels2.split('.')[nparam]
        print("\nBest fit parameters of REAL 2-parameter kernel = %s fit .... "\
              %options.kernels2.split('.')[nparam])
        qv2.printFitParams()
        # print "        Chi2/dof = %.7f +/- %.7f"%(meanc,np.sqrt(dumc))


        # Add valence pdf fits to plots
        qv2.plotPDF(ax_pdf_qv)
        if options.showPDFInset:
            qv2.plotPDF(ax_qv_pdf_inset)
        qv2.plotXPDF(ax_xpdf_qv)

        # Only add real ITD fit if kernel is C
        if qv2.kernel == 'C':
            qv2.plotITDReal(ax_itd_real_and_fit)
            if options.writeITDsFromFits:
                qv2.writeITD_RE()
                qv2.writePDF()

        # # Add Fit Results to PDF and xPDF Plots
        # qv2.addParamsToPlot(ax_pdf_qv,buff)
        # qv2.addParamsToPlot(ax_xpdf_qv,buff)
        # buff+=1

####################################################################
####################################################################
# PROCEED IF WE HAVE A 2-PARAMETER IMAG FIT
####################################################################
####################################################################
if options.fit2ParamsPlus != "":
    # Loop over all 2-type imag fit results
    for nparam, param2File in enumerate(options.fit2ParamsPlus.split('@')):

        # Reset the fit parameter dictionaries
        fitParamsAll2P={'chi2': [], 'normP': [], 'alphaP': [], 'betaP': []}
        params2=['chi2', 'normP', 'alphaP', 'betaP']

        # Read fit parameters from this param2File
        with open(param2File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                for n, k in enumerate(params2):
                    fitParamsAll2P[k].append(float(L[n]))

        # Get the chi2/dof
        dumc=0.0
        meanc=np.mean(fitParamsAll2P['chi2'])
        for g in range(0,gauge_configs):
            dumc+=np.power(fitParamsAll2P['chi2'][g]-meanc,2)
        dumc*=((1.0*(gauge_configs-1))/gauge_configs)

        localColors=['tomato','indigo']
        # Instantiate a 2-parameter qplus pdf
        qp2=PDF.pdf2('+',options.kernels2.split('.')[nparam],localColors[nparam],\
                     {'chi2': fitParamsAll2P['chi2'], 'alpha': fitParamsAll2P['alphaP'],\
                      'beta': fitParamsAll2P['betaP'],'norm': fitParamsAll2P['normP']},\
                     bool(options.corrSwitch),gauge_configs)
        qp2.parse()
        qp2.quarkSum() # check normalization
        # Dump fit param covariances determined from parsing of jackknife data
        print("2-QPLUS Covariance")
        qp2.printFitCov()
        qp2.printNorm()

        print("\nBest fit parameters of IMAG 2-parameter kernel = %s fit .... "\
              %options.kernels2.split('.')[nparam])
        qp2.printFitParams()
        print("        Chi2/dof = %.7f +/- %.7f"%(meanc,np.sqrt(dumc)))
        
        # Add qplus pdf fits to plots
        qp2.plotPDF(ax_pdf_qplus)
        if options.showPDFInset:
            qp2.plotPDF(ax_qplus_pdf_inset)
        qp2.plotXPDF(ax_xpdf_qplus)

        # Only add imag ITD fit if kernel is C
        if qp2.kernel == 'C':
            qp2.plotITDImag(ax_itd_imag_and_fit)
            if options.writeITDsFromFits:
                qp2.writeITD_IM()
                qp2.writePDF()

        # # Add Fit Results to PDF and xPDF Plots
        # qp2.addParamsToPlot(ax_pdf_qplus,buff)
        # qp2.addParamsToPlot(ax_xpdf_qplus,buff)
        # buff+=1        
####################################################################
####################################################################


####################################################################
####################################################################
# PROCEED IF WE HAVE A 3-PARAMETER REAL FIT
####################################################################
####################################################################
if options.fit3ParamsValence != "":
    # Loop over all 3-type real fit results
    for nparam, param3File in enumerate(options.fit3ParamsValence.split('@')):

        # Reset the fit parameter dictionaries
        fitParamsAll3V={'chi2': [], 'alpha': [], 'beta': [], 'delta': []}
        params3=['chi2', 'alpha', 'beta', 'delta']

        # Read fit parameters from this param3File
        with open(param3File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                # As of 02-23-2021 Fit params include norm, which is not needed in qval, so remove it
                # As of 04-10-2021 Fit params always include <chi> <norm> <a> <b> <g> <d>
                # ...so remove <g> entry as well, which is fixed to zero for 3-param fits
                del L[1]
                del L[3] # w/ L[1] already deleted, must remove L[3] instead of L[4]
                for n, k in enumerate(params3):
                    fitParamsAll3V[k].append(float(L[n]))

        # Get the chi2/dof
        dumc=0.0
        meanc=np.mean(fitParamsAll3V['chi2'])
        for g in range(0,gauge_configs):
            dumc+=np.power(fitParamsAll3V['chi2'][g]-meanc,2)
        dumc*=((1.0*(gauge_configs-1))/gauge_configs)
                
        # Instantiate a 3-parameter valence pdf
        qv3=PDF.pdf3('v',options.kernels3.split('.')[nparam],mainColors[nparam+2],\
                     {'chi2': fitParamsAll3V['chi2'], 'alpha': fitParamsAll3V['alpha'],\
                      'beta': fitParamsAll3V['beta'],\
                      'delta': fitParamsAll3V['delta'],'norm': None},bool(options.corrSwitch),\
                     gauge_configs)
        qv3.parse()
        qv3.quarkSum() # check normalization
        # Dump fit param covariances determined from parsing of jackknife data
        print("3-VALENCE Covariance")
        qv3.printFitCov()
        qv3.printNorm()

        print("\nBest fit parameters of REAL 3-parameter kernel = %s fit .... "\
              %options.kernels3.split('.')[nparam])
        # print "\nBest fit parameters of REAL 3-parameter kernel = %s fit .... "\
        #     %options.kernels3.split('.')[nparam]
        qv3.printFitParams()
        print("        Chi2/dof = %.7f +/- %.7f"%(meanc,np.sqrt(dumc)))


        # Add pdf fits to plots
        qv3.plotPDF(ax_pdf_qv)
        if options.showPDFInset:
            qv3.plotPDF(ax_qv_pdf_inset)
        qv3.plotXPDF(ax_xpdf_qv)

        # Only add real ITD fit if kernel is C
        if qv3.kernel == 'C':
            qv3.plotITDReal(ax_itd_real_and_fit)
            if options.writeITDsFromFits:
                qv3.writeITD_RE()
                qv3.writePDF()

        # # Add Fit Results to PDF and xPDF Plots
        # qv3.addParamsToPlot(ax_pdf_qv,buff)
        # qv3.addParamsToPlot(ax_xpdf_qv,buff)
        # buff+=1

####################################################################
####################################################################
# PROCEED IF WE HAVE A 3-PARAMETER IMAG FIT
####################################################################
####################################################################
if options.fit3ParamsPlus != "":
    # Loop over all 3-type imag fit results
    for nparam, param3File in enumerate(options.fit3ParamsPlus.split('@')):

        # Reset the fit parameter dictionaries
        fitParamsAll3P={'chi2': [], 'normP': [], 'alphaP': [], 'betaP': [], 'deltaP': []}
        params3=['chi2', 'normP', 'alphaP', 'betaP', 'deltaP']

        # Read fit parameters from this param3File
        with open(param3File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                # As of 04-10-2021 Fit params always include <chi> <norm> <a> <b> <g> <d>
                # ...so remove <g> entry as well, which is fixed to zero for 3-param fits
                del L[4]
                for n, k in enumerate(params3):
                    fitParamsAll3P[k].append(float(L[n]))

        # Get the chi2/dof
        dumc=0.0
        meanc=np.mean(fitParamsAll3P['chi2'])
        for g in range(0,gauge_configs):
            dumc+=np.power(fitParamsAll3P['chi2'][g]-meanc,2)
        dumc*=((1.0*(gauge_configs-1))/gauge_configs)

        # Instantiate a 3-parameter qplus pdf
        qp3=PDF.pdf3('+','C','navy',\
                     {'chi2': fitParamsAll3P['chi2'], 'alpha': fitParamsAll3P['alphaP'],\
                      'beta': fitParamsAll3P['betaP'], 'delta': fitParamsAll3P['deltaP'],\
                      'norm': fitParamsAll3P['normP']},bool(options.corrSwitch),gauge_configs)
        qp3.parse()
        qp3.quarkSum() # check normalization
        # Dump fit param covariances determined from parsing of jackknife data
        print("3-QPLUS Covariance")
        qp3.printFitCov()
        qp3.printNorm()

        print("\nBest fit parameters of IMAG 3-parameter kernel = C fit .... ")#\
            # %options.kernels3.split('.')[nparam]
        qp3.printFitParams()
        print("        Chi2/dof = %.7f +/- %.7f"%(meanc,np.sqrt(dumc)))

        # Add pdf fits to plots
        qp3.plotPDF(ax_pdf_qplus)
        if options.showPDFInset:
            qp3.plotPDF(ax_qplus_pdf_inset)
        qp3.plotXPDF(ax_xpdf_qplus)

        # Only add imag ITD fit if kernel is C
        if qp3.kernel == 'C':
            qp3.plotITDImag(ax_itd_imag_and_fit)
            if options.writeITDsFromFits:
                qp3.writeITD_IM()
                qp3.writePDF()

        # # Add Fit Results to PDF and xPDF Plots
        # qp3.addParamsToPlot(ax_pdf_qplus,buff)
        # qp3.addParamsToPlot(ax_xpdf_qplus,buff)
        # buff+=1
####################################################################
####################################################################


####################################################################
####################################################################
# PROCEED IF WE HAVE A 4-PARAMETER REAL FIT
####################################################################
####################################################################
if options.fit4ParamsValence != "":
    # Loop over all 4-type real fit results
    for nparam, param4File in enumerate(options.fit4ParamsValence.split('@')):

        # Reset the fit parameter dictionaries
        fitParamsAll4V={'chi2': [], 'alpha': [], 'beta': [], 'gamma': [], 'delta': []}
        params4=['chi2', 'alpha', 'beta', 'gamma', 'delta']

        # Read fit parameters from this param4File
        with open(param4File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                # As of 02-23-2021 Fit params include norm, which is not needed in qval, so remove it
                del L[1]
                for n, k in enumerate(params4):
                    fitParamsAll4V[k].append(float(L[n]))

        # Get the chi2/dof
        dumc=0.0
        meanc=np.mean(fitParamsAll4V['chi2'])
        for g in range(0,gauge_configs):
            dumc+=np.power(fitParamsAll4V['chi2'][g]-meanc,2)
        dumc*=((1.0*(gauge_configs-1))/gauge_configs)

        # Instantiate a 4-parameter valence pdf
        qv4=PDF.jacobi4('v',options.kernels4.split('.')[nparam],mainColors[1],\
                        {'chi2': fitParamsAll4V['chi2'], 'alpha': fitParamsAll4V['alpha'],\
                         'beta': fitParamsAll4V['beta'], 'gamma': fitParamsAll4V['gamma'],\
                         'delta': fitParamsAll4V['delta'],'norm': None},bool(options.corrSwitch))
        qv4.parse()
        qv4.quarkSum() # check normalization
        # Dump fit param covariances determined from parsing of jackknife data
        print("4-VALENCE Covariance")
        qv4.printFitCov()
        qv4.printNorm()

        # print "\nBest fit parameters of REAL 4-parameter kernel = %s fit .... "\
        #     %options.kernels4.split('.')[nparam]
        print("\nBest fit parameters of REAL 4-parameter kernel = %s fit .... "\
              %options.kernels4.split('.')[nparam])
        qv4.printFitParams()
        print("        Chi2/dof = %.7f +/- %.7f"%(meanc,np.sqrt(dumc)))

        # Add pdf fits to plots
        qv4.plotPDF(ax_pdf_qv)
        if options.showPDFInset:
            qv4.plotPDF(ax_qv_pdf_inset)
        qv4.plotXPDF(ax_xpdf_qv)

        # Only add real ITD fit if kernel is C
        if qv4.kernel == 'C':
            qv4.plotITDReal(ax_itd_real_and_fit)
            if options.writeITDsFromFits:
                qv4.writeITD_RE()
                qv4.writePDF()

        # # Add Fit Results to PDF and xPDF Plots
        # qv4.addParamsToPlot(ax_pdf_qv,buff)
        # qv4.addParamsToPlot(ax_xpdf_qv,buff)
        # buff+=1
        
####################################################################
####################################################################
# PROCEED IF WE HAVE JACOBI APPROX OF PITD
####################################################################
####################################################################
for J in [(options.jacobiPITDValence,'-','Re',1), (options.jacobiPITDPlus,'+','Im',0)]: # N.B. IM SHOULD HAVE '0' instead of '1'!!!
    corrStart = J[3]
    if J[0] != "":
        for nparam, paramFile in enumerate(J[0].split('@')):

            numLT = int(options.jacobiCorrections.split('.')[0])
            numAZ = int(options.jacobiCorrections.split('.')[1])
            numT4 = int(options.jacobiCorrections.split('.')[2])
            numT6 = int(options.jacobiCorrections.split('.')[3])
            numT8 = int(options.jacobiCorrections.split('.')[4])
            numT10 = int(options.jacobiCorrections.split('.')[5])
            

            # Reset/form the fitParams dict & paramOrder list
            # fitParams={'chi2': [], '\\alpha': [], '\\beta': []}
            # paramOrder=['chi2', '\\alpha', '\\beta']
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
            
            
            print(len(paramOrder))
            # Read fit parameters from this paramFile
            with open(paramFile) as ptr:
                for cnt, line in enumerate(ptr):
                    L=line.split(' ')
                    for n, k in enumerate(paramOrder):
                        fitParams[k].append(float(L[n]))


            # Instantiate a pitdJacobi instance, w/ pdf + corrections in real pitd
            jacColors=['purple', 'orange', 'saddlebrown','darkblue','darkgreen','black']
            if options.darkBkgd:
                jacColors=['cyan', 'lightgray', cm.copper(255),'dodgerblue']

            
            jac=PDF.pitdJacobi(J[1],jacColors,\
                               {k: fitParams[k] for k,v in fitParams.items()},\
                               [numLT,numAZ,numT4,numT6,numT8,numT10],bool(options.corrSwitch),\
                               gauge_configs,J[2],corrStart,dirac=DIRAC)
            jac.parse()

            print("For %s Jacobi..."%J[2])
            jac.printFitParams()
            # print(jac.pitd(1.0,1))
            print(jac.pCov)
            # sys.exit()

            
            # Finally plot the PDFs from jacobi polynomial fits
            if J[2] == 'Re':
                if options.showPDFInset:
                    jac.plotPDFs(ax_qv_pdf_inset,True)
                jac.plotPDFs(ax_pdf_qv)
                jac.plotXPDFs(ax_xpdf_qv)
                jac.plotCovHMap(fig_re_cov_heat,ax_re_cov_heat,'coolwarm') #bwr')
                jac.plotITDs(ax_itd_real_and_fit)
            else:
                if options.showPDFInset:
                    jac.plotPDFs(ax_qplus_pdf_inset,True)
                jac.plotPDFs(ax_pdf_qplus)
                jac.plotXPDFs(ax_xpdf_qplus)
                jac.plotCovHMap(fig_im_cov_heat,ax_im_cov_heat,'coolwarm') #bwr')
                jac.plotITDs(ax_itd_imag_and_fit)


            perZRanges=[]
            # Plot each jacobi fit band for a given zsep\in{1,16} atop pitd data
            if DIRAC == 8:
                perZRanges=[0.95,0.85,0.65,0.4,0.2,0,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
            if DIRAC == 11:
                # perZRanges=[0.94,0.75,0.55,0.3,0.2,0,-0.3,-0.3] # w/o SVD
                perZRanges=[0.925,0.74,0.5,0.3,0.12,-0.1,-0.3,-0.3] # w/ SVD
                
            for z, ax in enumerate(ax_pitd_re_perz.flat):
                ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
                ax.set_ylim(perZRanges[z],1)
                # jac.plotPITD(ax,z+1,dispColors[z+1])
                jac.plotPITD(ax,z+1,\
                             col=dispColors[z+1] if (z+1 >= int(options.h5ZCut.split('.')[0])\
                                                   and z+1 <= int(options.h5ZCut.split('.')[1]))\
                             else 'gray')

                PDF.h5Plot(options.pITD,options.h5ZCut,options.h5PCut,[ax],\
                           [None],gauge_configs,'pitd',dispColors,\
                           dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy,\
                           showZSepOnPlot=True,sysErrH5File=sysErrPITD,\
                           plotSingleZ=z+1)

            if DIRAC == 8:
                perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7,0.75,0.8,1.1,1.1]
            if DIRAC == 11:
                # perZRanges=[-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
                # perZRanges=[0.2,0.4,0.55,0.65,0.7,0.75,0.7,0.7]
                perZRanges=[0.28,0.51,0.7,0.8,0.8,0.75,0.72,0.7]                
                
            for z, ax in enumerate(ax_pitd_im_perz.flat):
                ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
                ax.set_ylim(0,perZRanges[z])
                # jac.plotPITD(ax,z+1,dispColors[z+1])
                jac.plotPITD(ax,z+1,\
                             col=dispColors[z+1] if (z+1 >= int(options.h5ZCut.split('.')[0])\
                                                   and z+1 <= int(options.h5ZCut.split('.')[1]))\
                             else 'gray')
                PDF.h5Plot(options.pITD,options.h5ZCut,options.h5PCut,[None],\
                           [ax],gauge_configs,'pitd',dispColors,\
                           dirac=DIRAC,oldH5Hierarchy=options.h5Hierarchy,\
                           showZSepOnPlot=True,sysErrH5File=sysErrPITD,\
                           plotSingleZ=z+1)





####
# Add the derived q/qbar distributions if qplus fit is provided
####
if options.fit2ParamsPlus != '':
    # Let's try the quark distribution
    Q2=PDF.Q2('','C','m',\
              {'alpha': fitParamsAll2V['alpha'],\
               'beta': fitParamsAll2V['beta'],\
               'norm': fitParamsAll2P['normP'],\
               'alphaP': fitParamsAll2P['alphaP'],\
               'betaP': fitParamsAll2P['betaP']},bool(options.corrSwitch),\
              gauge_configs)
    Q2.parse()
    Q2.quarkSum()
    Q2.plotPDF(ax_pdf_q)
    # Q2.plotXPDF(ax_xpdf)

    # Let's try the sea quark distribution
    QBar2=PDF.QBar2('bar','C','y',\
                    {'alpha': fitParamsAll2V['alpha'],\
                     'beta': fitParamsAll2V['beta'],\
                     'norm': fitParamsAll2P['normP'],\
                     'alphaP': fitParamsAll2P['alphaP'],\
                     'betaP': fitParamsAll2P['betaP']},bool(options.corrSwitch),\
                    gauge_configs)
    QBar2.parse()
    QBar2.quarkSum()
    QBar2.plotPDF(ax_pdf_qbar)
    # QBar2.plotXPDF(ax_xpdf)


            

# Set labels of figures
pdfAxName=''
if options.insertion == 8: pdfAxName='f'
if options.insertion == 11: pdfAxName='g'

for a in [ax_pdf_qv, ax_pdf_qplus, ax_pdf_q, ax_pdf_qbar]:
    a.set_xlabel(r'$x$')
ax_pdf_qv.set_ylabel(r'$%s_{q_-/N}\left(x,\mu^2\right)$'%pdfAxName)
ax_pdf_qplus.set_ylabel(r'$%s_{q_+/N}\left(x,\mu^2\right)$'%pdfAxName)
ax_pdf_q.set_ylabel(r'$%s_{q/N}\left(x,\mu^2\right)$'%pdfAxName)
ax_pdf_qbar.set_ylabel(r'$%s_{\bar{q}/N}\left(x,\mu^2\right)$'%pdfAxName)
for a in [ax_xpdf_qv, ax_xpdf_qplus]:
    a.set_xlabel(r'$x$')
ax_xpdf_qv.set_ylabel(r'$x%s_{q_-/N}\left(x,\mu^2\right)$'%pdfAxName)
ax_xpdf_qplus.set_ylabel(r'$x%s_{q_+/N}\left(x,\mu^2\right)$'%pdfAxName)
    

# Fetch all the unique labels on each fig, and make the legends
for fig, ax in [(fig_pdf_qv,ax_pdf_qv), (fig_pdf_qplus,ax_pdf_qplus),\
                (fig_pdf_q,ax_pdf_q), (fig_pdf_qbar,ax_pdf_qbar),\
                (fig_xpdf_qv,ax_xpdf_qv), (fig_xpdf_qplus,ax_xpdf_qplus),\
                (fig_pitd_real,ax_pitd_real), (fig_pitd_re_perz,ax_pitd_re_perz),\
                (fig_pitd_imag,ax_pitd_imag), (fig_pitd_im_perz,ax_pitd_im_perz),\
                (fig_itd_real,ax_itd_real), (fig_itd_imag,ax_itd_imag),\
                (fig_itd_real_and_fit,ax_itd_real_and_fit),\
                (fig_itd_imag_and_fit,ax_itd_imag_and_fit)]:

# for fig, ax in [(fig_itd_imag_and_fit,ax_itd_imag_and_fit)]:

    try:
        # Get the handles/labels
        handles, labels = fig.axes[0].get_legend_handles_labels()
        # handles, labels = fig.gca().get_legend_handles_labels()
        Ncol=2
        
        # If dealing with pdf figures, handle legends differently
        if ax == ax_pdf_qv or ax == ax_pdf_qplus or ax == ax_pdf_q or ax == ax_pdf_qbar\
           or ax == ax_xpdf_qv or ax == ax_xpdf_qplus:
            
            if ax == ax_pdf_qv or ax == ax_pdf_qplus or ax == ax_xpdf_qv or ax == ax_xpdf_qplus:
                Ncol=1

            dumH=[]
            dumL=[]
            numLabels=len(labels)
            halfNumLabels=int(numLabels/2)
        
            for l in range(0,halfNumLabels):
                dumH.append( (handles[l],handles[halfNumLabels+l]) )
                dumL.append( labels[l] )
            customHandles=dumH
            customLabels=tuple(dumL)
        
            ax.legend(customHandles,customLabels,framealpha=LegendFrameAlpha,\
                      ncol=Ncol)

        # Need something fancier for ITD figs that include lots of data + fits
        else:
            # First occurence of a displacement
            catch=labels.index(r'$z=%s$'%options.h5ZCut.split('.')[0])


            labelsNew=[]
            handlesNew=[]
            for n in range(0,int(catch/2)):
                labelsNew.append(labels[n])
                handlesNew.append( (handles[n],handles[n+int(catch/2)]) )

            # Tack on the remaining labels/handles
            for r,L in enumerate(labels[catch:]):
                labelsNew.append(L)
                handlesNew.append(handles[r])
            labels=labelsNew
            handles=handlesNew
            
            
            by_label = OrderedDict(zip(labels, handles))
            

            legend = ax.legend(by_label.values(), by_label.keys(),\
                               framealpha=LegendFrameAlpha,ncol=Ncol)# ,bbox_to_anchor=(0.9,1))
            for t in legend.get_texts():
                t.set_ha('left') # ha is alias for horizontalalignment
                t.set_position((shift,0))
            frame = legend.get_frame()
            frame.set_facecolor(legendFaceColor)
    except:
        continue
# sys.exit()


jacobiFigPrefix=''
if options.jacobiPITDValence != "" or options.jacobiPITDPlus != "":
    jacobiFigPrefix="jacobi_%slt_%saz_%st4_%st6_%st8_%st10.pmin%s_pmax%s_zmin%s_zmax%s."%\
        (options.jacobiCorrections.split('.')[0],options.jacobiCorrections.split('.')[1],\
         options.jacobiCorrections.split('.')[2],options.jacobiCorrections.split('.')[3],\
         options.jacobiCorrections.split('.')[4],options.jacobiCorrections.split('.')[5],\
         options.h5PCut.split('.')[0],options.h5PCut.split('.')[1],\
         options.h5ZCut.split('.')[0],options.h5ZCut.split('.')[1])



fig_pdf_qv.savefig(jacobiFigPrefix+"QV_PDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                   dpi=400,transparent=truthTransparent,\
                   bbox_inches='tight',pad_inches=0.1,format=form)
fig_xpdf_qv.savefig(jacobiFigPrefix+"QV_xPDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                    dpi=400,transparent=truthTransparent,\
                    bbox_inches='tight',pad_inches=0.1,format=form)
fig_itd_real_and_fit.savefig(jacobiFigPrefix+"itdReal_andfit_as%.4f%s.%s"\
                             %(options.alphaS,suffix,form),dpi=400,transparent=truthTransparent,\
                             bbox_inches='tight',pad_inches=0.1,format=form)
fig_itd_imag_and_fit.savefig(jacobiFigPrefix+"itdImag_andfit_as%.4f%s.%s"\
                             %(options.alphaS,suffix,form),dpi=400,transparent=truthTransparent,\
                             bbox_inches='tight',pad_inches=0.1,format=form)


fig_pdf_qplus.savefig(jacobiFigPrefix+"Q+_PDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                      dpi=400,transparent=truthTransparent,bbox_inches='tight',\
                      pad_inches=0.1,format=form)
fig_xpdf_qplus.savefig(jacobiFigPrefix+"Q+_xPDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                       dpi=400,transparent=truthTransparent,\
                       bbox_inches='tight',pad_inches=0.1,format=form)
fig_pdf_q.savefig(jacobiFigPrefix+"Q_PDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                  dpi=400,transparent=truthTransparent,\
                  bbox_inches='tight',pad_inches=0.1,format=form)
fig_pdf_qbar.savefig(jacobiFigPrefix+"QBAR_PDF_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                     dpi=400,transparent=truthTransparent,\
                     bbox_inches='tight',pad_inches=0.1,format=form)


if options.pITD != "":
    fig_pitd_real.savefig(jacobiFigPrefix+"pitdReal_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                          dpi=400,transparent=truthTransparent,\
                          bbox_inches='tight',pad_inches=0.1,format=form)
    fig_pitd_imag.savefig(jacobiFigPrefix+"pitdImag_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                          dpi=400,transparent=truthTransparent,\
                          bbox_inches='tight',pad_inches=0.1,format=form)

    fig_pitd_re_perz.savefig(jacobiFigPrefix+"pitd_re_perz_as%.4f%s.%s"\
                             %(options.alphaS,suffix,form),dpi=400,\
                             transparent=truthTransparent,\
                             bbox_inches='tight',pad_inches=0.1,format=form)
    fig_pitd_im_perz.savefig(jacobiFigPrefix+"pitd_im_perz_as%.4f%s.%s"\
                             %(options.alphaS,suffix,form),dpi=400,\
                             transparent=truthTransparent,\
                             bbox_inches='tight',pad_inches=0.1,format=form)
    fig_re_cov_heat.savefig(jacobiFigPrefix+"jacTexp_pitd_re_covheat_as%.4f%s.%s"\
                            %(options.alphaS,suffix,form),dpi=400,\
                            transparent=truthTransparent,bbox_inches='tight',\
                            pad_inches=0.1,format=form)
    fig_im_cov_heat.savefig(jacobiFigPrefix+"jacTexp_pitd_im_covheat_as%.4f%s.%s"\
                            %(options.alphaS,suffix,form),dpi=400,\
                            transparent=truthTransparent,bbox_inches='tight',\
                            pad_inches=0.1,format=form)
    
if options.msbarITD != "":
    fig_itd_real.savefig(jacobiFigPrefix+"itdReal_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                         dpi=400,transparent=truthTransparent,\
                         bbox_inches='tight',pad_inches=0.1,format=form)
    fig_itd_imag.savefig(jacobiFigPrefix+"itdImag_as%.4f%s.%s"%(options.alphaS,suffix,form),\
                         dpi=400,transparent=truthTransparent,\
                         bbox_inches='tight',pad_inches=0.1,format=form)

plt.show()
