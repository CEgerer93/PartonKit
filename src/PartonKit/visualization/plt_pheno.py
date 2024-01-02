#!/usr/bin/python3

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
lhapdf.pathsPrepend("/home/colin/LHAPDF-6.3.0/share/LHAPDF/LHAPDF")

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-Q", "--phenoQ", type="int", default=1,
                  help='Evaluate pheno pdfs at <phenoQ GeV> (default = 1)')
parser.add_option("-d", "--dist", type="str", default='',
                  help='Quark PDFs to show: <q,qv,qbar,q+> (default = "")')
parser.add_option("-i", "--showPDFInset", type="int", default=0,
                  help='Make PDF inset visible (default = 0)')
parser.add_option("-f", "--pdfFit", default='x.x',
                  help='Text file of AIC results stored as (x,avg,err) per line; expects <qv AIC>:<q+ AIC> (default ="x.x")')
parser.add_option("--darkBkgd", action="store_true", default=False, dest="darkBkgd",
                  help='Format figs for dark background (default = False)')


# Parse the input arguments
(options, args) = parser.parse_args()

DIRAC=options.insertion
dist=options.dist

phenoPDFs=pheno.pheno(options.phenoQ,DIRAC,options.darkBkgd)
phenoPDFs.accessPDFs()
phenoPDFs.extractPDFs()
            
                

###########
# Instantiate figure
###########
fig_pdf=plt.figure(figsize=(10,5))
ax_pdf=fig_pdf.gca()
# Set ranges of figures
ax_pdf.set_xlim([0.0,1.0])
ax_pdf.set_ylim([-0.2,5])
# Change ylimits for qbar
if dist == 'qbar':
    ax_pdf.set_ylim([-1,3.5])
    
if options.showPDFInset:
    # ax_pdf_inset=fig_pdf.add_axes([0.28,0.55,0.35,0.3]) # percentages of original ax_pdf object
    ax_pdf_inset=fig_pdf.add_axes([0.29,0.55,0.338,0.29]) # percentages of original ax_pdf object
    ax_pdf_inset.set_xlim([0.51,1.0])
    ax_pdf_inset.set_ylim([-0.1,0.3])
    # Zoom in further for qbar
    if dist == 'qbar':
        ax_pdf_inset.set_ylim([-0.05,0.05])


        
distLabel='q' if dist == 'q' else '\overline{q}'
        
# Set labels of figures
pdfAxName=''; thisWorkLabel='';
if options.insertion == 8:
    pdfAxName='f'
    thisWorkLabel='$%s_{%s/N}^{\\rm AIC}\left(x,\mu^2\\right)$'%(pdfAxName,distLabel)
if options.insertion == 11:
    pdfAxName='g'
    thisWorkLabel='$%s_{%s/N}^{\\rm AIC}\left(x,\mu^2\\right)/g_A\left(\mu^2\\right)$'%(pdfAxName,distLabel)



ax_pdf.set_xlabel(r'$x$')
ax_pdf.set_ylabel(r'$%s_{%s/N}\left(x,\mu^2\right)$'%(pdfAxName,distLabel))



for phN, phK in enumerate(phenoPDFs.F):
    for AX in [ax_pdf, ax_pdf_inset]:
        AX.plot(phenoPDFs.phenoX,phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                color=phenoPDFs.F[phK]['color'],label=phenoPDFs.F[phK]['label'])
        AX.fill_between(phenoPDFs.phenoX,\
                        [a+e for a, e in zip(phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                                             phenoPDFs.F[phK]['pdf2plot'][dist]['errplus'])],\
                        [a-e for a, e in zip(phenoPDFs.F[phK]['pdf2plot'][dist]['central'],\
                                             phenoPDFs.F[phK]['pdf2plot'][dist]['errminus'])],\
                        color=phenoPDFs.F[phK]['color'],alpha=phenoPDFs.phenoAlpha,\
                        label=phenoPDFs.F[phK]['label'],hatch=phenoPDFs.phenoHatch,lw=0)

        
# Optionally include provided PDF file
for AX in [ax_pdf, ax_pdf_inset]:
    if options.pdfFit != "":
        a=[]; e=[]

        x1=[]; a1=[]; e1=[]
        x2=[]; a2=[]; e2=[]

        for nf,fit in enumerate(options.pdfFit.split(':')):
            
            with open(fit) as ptr:
                for cnt, line in enumerate(ptr):
                    L=line.rstrip().split(' ')
                    
                    if L[1] == 'inf' or L[2] == 'nan':
                        continue
                    else:
                        if nf == 0:
                            x1.append(float(L[0]))
                            a1.append(float(L[1]))
                            e1.append(float(L[2]))
                        else:
                            x2.append(float(L[0]))
                            a2.append(float(L[1]))
                            e2.append(float(L[2]))

                            
        # Assuming same x slices in pdf fit(s)
        a=[0.5*(f1+f2) for f1,f2 in zip(a1,a2)] if dist=='q' else\
            [0.5*(f2-f1) for f1,f2 in zip(a1,a2)]
        e=[np.sqrt(0.25*f1**2+0.25*f2**2) for f1,f2 in zip(e1,e2)]

        AX.plot(x1,a,color='purple',label=thisWorkLabel)
        AX.fill_between(x1,[ai+ei for ai,ei in zip(a,e)],[ai-ei for ai,ei in zip(a,e)],\
                        color='purple',lw=0,alpha=0.3,\
                        label=thisWorkLabel)


# Fetch all the unique labels on each fig, and make the legends
handles, labels = fig_pdf.axes[0].get_legend_handles_labels()
dumH=[]
dumL=[]
numLabels=len(labels)
halfNumLabels=int(numLabels/2)

for l in range(0,halfNumLabels):
    dumH.append( (handles[l],handles[halfNumLabels+l]) )
    dumL.append( labels[l] )
customHandles=dumH
customLabels=tuple(dumL)

ax_pdf.legend(customHandles,customLabels,framealpha=LegendFrameAlpha,fontsize=15) #19)

fig_pdf.savefig("phenoPDFs_%s.%s"%(dist,form),dpi=400,transparent=truthTransparent,\
                bbox_inches='tight',pad_inches=0.1,format=form)
plt.show()
