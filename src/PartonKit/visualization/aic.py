#!/usr/bin/python3.8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdf_utils as PDF
import sys,optparse
import os
import aic_utils
import pheno
from common_fig import *

import scipy.integrate as integrate

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-m", "--models", type="str",default="",
                  help='Colon delimited fit results of diff. models (default='')')
parser.add_option("-j", "--jacobiTruncs", type="str", default="",
                  help="Jacobi approx of pITD truncation orders;\n<numLT>.<numAZ>.<numT4>.<numT6>:etc (default = "")")
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-H", "--showPheno", type="int", default=0,
                  help='Show pheno PDFs (default=0)')
parser.add_option("-Q", "--phenoQ", type="int", default=1,
                  help='Evaluate pheno pdfs at <phenoQ GeV> (default = 1)')
parser.add_option("-i", "--showPDFInset", action="store_true", default=False, dest="showPDFInset",
                  help='Make PDF inset visible (default = 0)')
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")
parser.add_option("--pdfType", type="str", default='',
                  help="Types: <v,+,'',bar> (default = '')")
parser.add_option("--darkBkgd", action="store_true", default=False, dest="darkBkgd",
                  help='Format figs for dark background (default = False)')
parser.add_option("--showFig", action="store_true", default=False, dest="showFig",
                  help='Show figure (default = False)')


# Parse the input arguments
(options, args) = parser.parse_args()

gauge_configs=options.cfgs
DIRAC=options.insertion


#################################
# Optionally Bring in pheno PDFs
#################################
if options.showPheno:
    phenoPDFs=pheno.pheno(options.phenoQ,DIRAC,options.darkBkgd)
    phenoPDFs.accessPDFs()
    phenoPDFs.extractPDFs()
#--------------------------------------------------------------------


# Figure
fig_pdf=plt.figure(figsize=(12,10))
ax_pdf=fig_pdf.gca(); ax_pdf_inset=None
ax_pdf.set_xlim([0.0,1.0])
ax_pdf.set_ylim([-0.25,4])
# ax_pdf.set_ylim([-0.5,6])
ax_pdf.set_xlabel(r'$x$')

pdfAxName=''
if options.insertion == 8: pdfAxName='f'
if options.insertion == 11: pdfAxName='g'

pdfNameSuffix='' if options.insertion == 8 else '/g_A\left(\mu^2\\right)'

pdfTypeForPlot='-' if options.pdfType == 'v' else options.pdfType

ax_pdf.set_ylabel(r'$%s_{q_%s/N}\left(x,\mu^2\right)$'%(pdfAxName,pdfTypeForPlot))

# Potentially add inset
if options.showPDFInset:
    # ax_pdf_inset=fig_pdf.add_axes([0.28,0.55,0.35,0.3]) # percentages of original ax_pdf object
    ax_pdf_inset=fig_pdf.add_axes([0.55,0.3,0.292,0.25]) # percentages of original ax_pdf object
    ax_pdf_inset.set_xlim([0.51,1.0])
    ax_pdf_inset.set_ylim([-0.1,0.3])



#############################
# Add pheno PDFs to pdf plot
#############################
if options.showPheno:
    for phN, phK in enumerate(phenoPDFs.F):

        axesToIncludePhenos=[(ax_pdf,'q%s'%options.pdfType)]
        if options.showPDFInset:
            axesToIncludePhenos.append((ax_pdf_inset,'q%s'%options.pdfType))
        
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
#-----------------------------------------------------------------------------------------------------

    
    

#####
MODELS=[]; NDATA=[]; CUTS=[]
# Read in all the models
for n,M in enumerate(options.models.split(':')):
    name="Model-%i"%n
    corrStart=0
    numLT,numAZ,numT4,numT6,numT8,numT10 = tuple(int(t) for t in options.jacobiTruncs.split(':')[n].split('.'))
    
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
        
        
    # Read fit parameters from this paramFile
    with open(M) as ptr:
        for cnt, line in enumerate(ptr):
            L=line.split(' ')
            for n, k in enumerate(paramOrder):
                fitParams[k].append(float(L[n]))

    
    # Instantiate a pitdJacobi instance, w/ pdf + corrections in real pitd
    jacColors=['purple', 'orange', 'saddlebrown','darkblue']
    if options.darkBkgd:
        jacColors=['cyan', 'lightgray', cm.copper(255),'dodgerblue']
        
        
    jac=PDF.pitdJacobi(name,jacColors,\
                       {k: fitParams[k] for k,v in fitParams.items()},\
                       [numLT,numAZ,numT4,numT6,numT8,numT10],True,\
                       gauge_configs,'Re',0,dirac=DIRAC)
    jac.parse()
    # jac.plotPDFs(ax_pdf)

    # Append jac to MODELS list
    MODELS.append(jac)

    # Determine how many datapts were fit from fit param file name
    print(M)
    pm,px,zm,zx=M.find('pmin'),M.find('pmax'),M.find('zmin'),M.find('zmax')
    # Append (pmax-pmin+1)*(zmax-zmin+1)
    NDATA.append( (int(M[px+4])-int(M[pm+4])+1)*(int(M[zx+4])-int(M[zm+4])+1) ) # super hacky!
    CUTS.append((int(M[pm+4]),int(M[px+4]),int(M[zm+4]),int(M[zx+4])))

# Instantiate an AIC object
aic=aic_utils.AIC(NDATA,MODELS,CUTS)
# Compute AIC average/error of Leading-twist PDF
aic.avgAIC()
aic.errAIC()
# Compute AIC average/error of Discretization effect
aic.avgAIC_AZ()
aic.errAIC_AZ()
# Compute AIC average/error of all Higher-twist effects
aic.avgAIC_HT()
aic.errAIC_HT()


fig_weights = plt.figure(figsize=(12,10))
ax_weights = fig_weights.gca()
ax_weights.hist(aic.aicWeights, bins=np.linspace(0,1,201), log=True)
ax_weights.set_ylabel(r'${\rm Yield}$')
ax_weights.set_xlabel(r'$w_i$')

# print(len(aic.aicWeights))


# Pair each model with its AIC weight
modelsAndWeights=dict(zip(aic.aicWeights,aic.models))
# Rip out the weights into an unsorted list
unsortWeights=[w for w in modelsAndWeights.keys()]


# Get coordinates of max weight
sortWeights = np.sort(unsortWeights)

# Sort on AIC weights and print
for n,W in enumerate(sortWeights[-5:]):
    print("For model %s, weight = %.5f"%(modelsAndWeights[W],W))
    print("              cuts = %i,%i,%i,%i"%(aic.modelsAndCuts[modelsAndWeights[W]]))

    # x,y coords of highest weight (plus small offsets)
    # xoffset = sortWeights[-1] #+0.02
    xoffset = 0.6-(5-n)*0.1 #0.05
    yshift=(5-n)*1.5
    downShift=yshift-(5-n)*0.07

    # Display the model
    # ax_weights.text(xoffset,yshift,s=r'$%s$'%modelsAndWeights[W].pNums,rotation=0,fontsize=18)


    thisModel = ax_weights.text(xoffset,yshift,s=r'$[%i,%i,%i,%i]\ p_{\rm min}=%i, p_{\rm max}=%i, z_{\rm min}=%i, z_{\rm max}=%i$'%(modelsAndWeights[W].pNums[0],modelsAndWeights[W].pNums[1],modelsAndWeights[W].pNums[2],modelsAndWeights[W].pNums[3],aic.modelsAndCuts[modelsAndWeights[W]][0],aic.modelsAndCuts[modelsAndWeights[W]][1],aic.modelsAndCuts[modelsAndWeights[W]][2],aic.modelsAndCuts[modelsAndWeights[W]][3]),fontsize=15)

    # Add line under printed model
    rangey=ax_weights.get_ylim()[1]-ax_weights.get_ylim()[0]
    ax_weights.plot([xoffset,xoffset+0.48],\
                    [downShift,downShift],color='gray')
    # ax_weights.plot([xoffset,xoffset+0.5],\
    #                 [yshift+(5-n)*np.exp(0.02)-np.exp(0.1),\
    #                  yshift+(5-n)*np.exp(0.02)-np.exp(0.1)],color='gray')
    
    # Add connector between histo bin and model
    # ...except for n=4 (i.e. highest weighted model)
    if True: # n != 4:
        lineX=[W,xoffset+0.1]
        lineY=[1,downShift]
        ax_weights.plot(lineX,lineY,ls='--',color='gray')





for a in [ax_pdf, ax_pdf_inset]:
    if type(a) != None:
        a.plot(aic.x,aic.aicAvg,color='purple',\
               label=r'$%s^{\rm AIC}_{q_%s/N}\left(x,\mu^2\right)%s$'%\
               (pdfAxName,pdfTypeForPlot,pdfNameSuffix))
        # Assuming all MODELS are of same generic type (eg. jacobi to real rpitd)
        a.fill_between(aic.x[:],aic.aicAvg[:]+aic.aicErr[:],
                       aic.aicAvg[:]-aic.aicErr[:],color='purple',alpha=0.3,lw=0,\
                       label=r'$%s^{\rm AIC}_{q_%s/N}\left(x,\mu^2\right)%s$'%\
                       (pdfAxName,pdfTypeForPlot,pdfNameSuffix))

        # Add discretization effect to plot
        a.plot(aic.x,aic.aicAvg_AZ,color='orange',\
               label=r'$\mathcal{O}(a/\left|z\right|)^{\rm AIC}$')
        # Assuming all MODELS are of same generic type (eg. jacobi to real rpitd)
        a.fill_between(aic.x[:],aic.aicAvg_AZ[:]+aic.aicErr_AZ[:],
                       aic.aicAvg_AZ[:]-aic.aicErr_AZ[:],color='orange',alpha=0.3,lw=0,\
                       label=r'$\mathcal{O}(a/\left|z\right|)^{\rm AIC}$')

        # Add Higher-twist effect to plot
        a.plot(aic.x,aic.aicAvg_HT,color='saddlebrown',\
               label=r'$\mathcal{O}(z^{2n}\Lambda_{\rm QCD}^{2n})^{\rm AIC}$')
        # Assuming all MODELS are of same generic type (eg. jacobi to real rpitd)
        a.fill_between(aic.x[:],aic.aicAvg_HT[:]+aic.aicErr_HT[:],
                       aic.aicAvg_HT[:]-aic.aicErr_HT[:],color='saddlebrown',alpha=0.3,lw=0,\
                       label=r'$\mathcal{O}(z^{2n}\Lambda_{\rm QCD}^{2n})^{\rm AIC}$')


# Write to output
OUT="AIC-SLICED_Q%s_PDF_as%.4f.txt"%(pdfTypeForPlot,options.alphaS)
aic.out(OUT)

# Manage legend
# handles, labels = fig_pdf.axes[0].get_legend_handles_labels()
handles, labels = ax_pdf.get_legend_handles_labels()
dumH=[]
dumL=[]
numLabels=len(labels)
halfNumLabels=int(numLabels/2)

for l in range(0,halfNumLabels):
    dumH.append( (handles[l],handles[halfNumLabels+l]) )
    dumL.append( labels[l] )
customHandles=dumH
customLabels=tuple(dumL)

ax_pdf.legend(customHandles,customLabels,framealpha=LegendFrameAlpha)
        

fig_pdf.savefig("AIC_Q%s_PDF_as%.4f%s.%s"%(pdfTypeForPlot,options.alphaS,suffix,form),dpi=400,\
                transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)

fig_weights.savefig("AIC_WEIGHTS_Q%s_PDF_as%.4f%s.%s"%(pdfTypeForPlot,options.alphaS,suffix,form),\
                    dpi=400,\
                    transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)

if options.showFig:
    plt.show()
