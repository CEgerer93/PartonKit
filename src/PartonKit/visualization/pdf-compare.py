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

parser.add_option("-N", "--normalizingForm", type="str", default="",
                  help='Normalizing PDF fit parameters file (default = "")')
parser.add_option("-n", "--normPZ_ranges", type="str", default="x.x_x.x",
                  help='[Norm] P & Z ranges of normalizing form: <pmin>.<pmax>_<zmin>.<zmax> (default = "x.x_x.x")')
parser.add_option("-A", "--altForm", type="str", default="x@x",
                  help='Alt PDF fit parameters file (default = "x@x")')
parser.add_option("-a", "--altPZ_ranges", type="str", default="x.x_x.x/y.y_y.y",
                  help='[Alt] P & Z ranges of alternate pdf forms: <pmin>.<pmax>_<zmin>.<zmax>/... (default = "x.x_x.x/y.y_y.y")')


# parser.add_option("-n", "--funcType", type="str", default="",
#                   help='Functional form of normalizing PDF (default = "")')
# parser.add_option("-k", "--norm_jacobiCorrections", type="str", default="",
#                   help='[Norm] Jacobi approx of pITD corrections; <numLT>.<numAZ>.<numT4>.<numT6> (default = "")')
# parser.add_option("-A", "--altForms", type="str", default="_@_",
#                   help='Alternate PDF forms to normalize (default = "_@_")')
# parser.add_option("-a", "--altFuncTypes", type="str", default="x.x",
#                   help='Functional forms of alternate PDFs (default = "x.x")')

parser.add_option("-j", "--jacobiCorrections", type="str", default="",
                  help='Jacobi approx of pITD corrections; <numLT>.<numAZ>.<numT4>.<numT6> (default = "")')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("--comp", type="str", default='x',
                  help='Component of pitd (Re (v) -or- Im (+)) (default = x)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("--logx", action="store_true", default=False, dest='logx',
                  help='Plot horizontal axis on a log scale')

# Parse the input arguments
(options, args) = parser.parse_args()

labelFontSize=14
gauge_configs=options.cfgs
DIRAC=options.insertion

# Color according the z value
# mainColors=[ cm.cividis(d) for d in np.linspace(0.1,2,len(options.altPZ_ranges.split('/'))) ]
mainColors=[ cm.twilight(d) for d in np.linspace(0.3,0.77,len(options.altPZ_ranges.split('/'))) ]
hcolor='gray'


pdf='f' if DIRAC==8 else 'g'
pdfType='-' if options.comp=='Re' else '+'
comp=options.comp
corrStart=1 if comp=='Re' else 0

    
###############
# INSTANTIATE FIGURES
###############
fig=plt.figure(figsize=(12,10))
ax=fig.gca()
# twist4=plt.figure(figsize=(12,10))
# ax_t4=twist4.gca()
# twist6=plt.figure(figsize=(12,10))
# ax_t6=twist6.gca()


# plt.subplots_adjust(hspace=0.0075)

# twist4, (ax_t4, ax_t4_chi) = plt.subplots(2,1,sharex=True,figsize=(12,10),\
#                                           gridspec_kw={'height_ratios': [5,2]})
# twist6, (ax_t6, ax_t6_chi) = plt.subplots(2,1,sharex=True,figsize=(12,10),\
#                                           gridspec_kw={'height_ratios': [5,2]})

twist, (ax_t4, ax_t6, blank, ax_twist_chi) = plt.subplots(4,1,sharex=False,figsize=(15,10),\
                                                          gridspec_kw={'height_ratios':\
                                                                       [3,3,1.5,2]})
plt.subplots_adjust(hspace=0)

blank.set_axis_off()

# ax_t4.xaxis.set_tick_params(labeltop=True,labelbottom=False)
# ax_t4.xaxis.tick_top()

ax_t4.set_xlim([0.75,4])
ax_t6.set_xlim([0.75,4])

ax_t4.set_xticks([]) # [1,2,3,4])
ax_t6.set_xlabel(r'$n$',labelpad=-2)
ax_t4.xaxis.set_label_position('top')
ax_t4.set_ylabel(r'$C^{%s}_n$'%'t4')

ax_t6.xaxis.set_ticks([1,2,3,4])

ax_t6.set_ylabel(r'$C^{%s}_n$'%'t6')

# ax_t4.set_xticks([1,2,3,4])
# Add a horizontal line at zero for reference
ax_t4.hlines(0,-5,10,hcolor,ls='--')
ax_t6.hlines(0,-5,10,hcolor,ls='--')


ax_twist_chi.set_ylabel(r'$\chi_r^2$')
ax_twist_chi.set_xlabel(r'$\left[p_{min},p_{max};z_{min},z_{max}\right]$')


# plt.show()
# sys.exit()



# # Associate funcTypes passed with pdf instances from pdf_utils
dict_pdfType={'jam2': {'ord': ['chi2','alpha','beta'],\
                       'params': {}, 'pdf': PDF.pdf2 },\
              'jam3': {'ord': ['chi2','alpha','beta','delta'],\
                       'params': {}, 'pdf': PDF.pdf3 },\
              'jam4': {'ord': ['chi2', 'alpha', 'beta', 'gamma', 'delta'],\
                       'params': {}, 'pdf': PDF.pdf4 },\
              'jacobi': {'ord': ['chi2', '\\alpha', '\\beta'],\
                         'params': {'chi2': [], '\\alpha': [], '\\beta': []},\
                         'pdf': PDF.pitdJacobi } }



def jacobiParamContainers(corrections,corrStart):
    
    numLT = int(corrections.split('.')[0])
    numAZ = int(corrections.split('.')[1])
    numT4 = int(corrections.split('.')[2])
    numT6 = int(corrections.split('.')[3])
    numT8 = int(corrections.split('.')[4])
    numT10 = int(corrections.split('.')[5])
    
    # Reset/form the params dict & order list
    # params={'chi2': [], '\\alpha': [], '\\beta': []}
    # order=['chi2', '\\alpha', '\\beta']
    params={'L2': [], 'chi2': [], 'L2/dof': [], 'chi2/dof': [], '\\alpha': [], '\\beta': []}
    order=['L2', 'chi2', 'L2/dof', 'chi2/dof', '\\alpha', '\\beta']
    for l in range(0,numLT):
        params.update({'C^{lt}_%d'%l: []})
        order.append('C^{lt}_%d'%l)
    for a in range(corrStart,numAZ+corrStart):
        params.update({'C^{az}_%d'%a: []})
        order.append('C^{az}_%d'%a)
    for t in range(corrStart,numT4+corrStart):
        params.update({'C^{t4}_%d'%t: []})
        order.append('C^{t4}_%d'%t)
    for s in range(corrStart,numT6+corrStart):
        params.update({'C^{t6}_%d'%s: []})
        order.append('C^{t6}_%d'%s)
    for u in range(corrStart,numT8+corrStart):
        fitParams.update({'C^{t8}_%d'%u: []})
        paramOrder.append('C^{t8}_%d'%u)
    for v in range(corrStart,numT10+corrStart):
        fitParams.update({'C^{t10}_%d'%v: []})
        paramOrder.append('C^{t10}_%d'%v)

    return params, order



# Collect the tick labels for chi2 subplots
tickLabelChi2=[]
# Associate a pdf handle with a file
fileMap={'ref': {'file': options.normalizingForm, 'ranges': options.normPZ_ranges}}
for n, pz in enumerate(options.altPZ_ranges.split('/')):

    file=options.altForm.split('@')[n]

    fileMap.update({str(n): {'file': file, 'ranges': pz}})
    
    # # Replace the PZ ranges from the ref file name
    # idx=fileMap['ref']['file'].find('pmin')
    # dum=fileMap['ref']['file']
    # # dum=fileMap['ref']['file'].replace(fileMap['ref']['file'][idx:],\
    # #                                    "pmin%s_pmax%s_zmin%s_zmax%s.txt"%\
    # #                                    (pz.split('_')[0].split('.')[0],\
    # #                                     pz.split('_')[0].split('.')[1],\
    # #                                     pz.split('_')[1].split('.')[0],\
    # #                                     pz.split('_')[1].split('.')[1]))
    

    # # Package the new alt pdf file name into fileMap dict
    # fileMap.update({str(n): {'file': dum, 'ranges': pz}})



    # Append tick labels for chi2 subplots
    tickLabelChi2.append(r'$\left[%s,%s;%s,%s\right]$'\
                              %(str(pz.split('_')[0].split('.')[0]),\
                                str(pz.split('_')[0].split('.')[1]),\
                                str(pz.split('_')[1].split('.')[0]),\
                                str(pz.split('_')[1].split('.')[1])))



ax_twist_chi.set_xticklabels(tickLabelChi2,size=18)



# Make a dictionary to hold all jacobi PDF parameterizations
allPDFs={}
for k in fileMap.keys():
    # Make empty dictionary and array to store parameters and order
    fitParams, paramOrder=jacobiParamContainers(options.jacobiCorrections,corrStart)
    # Fill them
    parse_util.reader(fileMap[k]['file'],fitParams,paramOrder,None)

    # Make the PDF
    dum=PDF.pitdJacobi(pdfType, ['blue', 'white', 'white', 'white'],\
                       {k: fitParams[k] for k,v in fitParams.items()},\
                       [int(n) for n in options.jacobiCorrections.split('.')],True,\
                       gauge_configs,comp,corrStart,dirac=DIRAC)

    dum.parse()
    print("For PDF = %s with P/Z ranges = %s"%(k,fileMap[k]['ranges']))
    dum.printFitParams()
    
    allPDFs.update({k: {'params': fitParams, 'order': paramOrder, 'pdf': dum,\
                        'ranges': fileMap[k]['ranges']} })



dum=[]
for xi in allPDFs['ref']['pdf'].x:
    dum.append(allPDFs['ref']['pdf'].pdfLT(xi))
# print(np.sort(dum))





    

# Iterate over all pdf forms to be normalized
for k,v in allPDFs.items():

    refPDF={'avg': [], 'errPlus': [], 'errMinus': []}
    normPDF={'avg': [], 'err': []}
    for xi in allPDFs['ref']['pdf'].x:

        if k == 'ref':
            refPDF['avg'].append(1)
            refPDF['errPlus'].append(1+(v['pdf'].pdfLTErr(xi)[0]/v['pdf'].pdfLT(xi)))
            refPDF['errMinus'].append(1-(v['pdf'].pdfLTErr(xi)[0]/v['pdf'].pdfLT(xi)))

        
        # normPDF['avg'].append(allPDFs['0']['pdf'].pdfLT(x)/allPDFs['ref']['pdf'].pdfLT(x))
        # normPDF['err'].append(allPDFs['0']['pdf'].pdfLTErr(x)[0]/allPDFs['ref']['pdf'].pdfLT(x))

        
        normPDF['avg'].append((1+v['pdf'].pdfLT(xi))/(allPDFs['ref']['pdf'].pdfLT(xi)+1))
        normPDF['err'].append(v['pdf'].pdfLTErr(xi)[0]/allPDFs['ref']['pdf'].pdfLT(xi))
        
    normErrUp=[a+e if ~np.isnan(a+e) else 1000 for a, e in zip(normPDF['avg'],normPDF['err'])]
    normErrLow=[a-e if ~np.isnan(a-e) else 1000 for a, e in zip(normPDF['avg'],normPDF['err'])]


    # Label for the normalized PDFs
    lab=r'$\left(%i,%i,%i,%i\right)\quad [%s,%s;%s,%s]$'\
        %(v['pdf'].numLT,v['pdf'].numAZ,v['pdf'].numHTs[0],v['pdf'].numHTs[1],\
          v['ranges'].split('_')[0].split('.')[0],\
          v['ranges'].split('_')[0].split('.')[1],\
          v['ranges'].split('_')[1].split('.')[0],\
          v['ranges'].split('_')[1].split('.')[1])
    
    # Plot the reference PDF
    if k == 'ref':
        ax.plot(allPDFs['ref']['pdf'].x,refPDF['avg'],color='gray',label=lab)
        ax.fill_between(allPDFs['ref']['pdf'].x,refPDF['errPlus'],refPDF['errMinus'],\
                        color='gray',alpha=0.3,lw=0,label=lab)
        continue
    # Plot the normalized PDFs
    else:
        
        
        
        # Plot the normalized PDF
        ax.plot(allPDFs['ref']['pdf'].x,normPDF['avg'],color=mainColors[int(k)],\
                label=lab)
        ax.fill_between(allPDFs['ref']['pdf'].x,normErrUp,normErrLow,color=mainColors[int(k)],\
                        alpha=0.3,lw=0,label=lab)
        
    


        
    
    # Now add the actual expansion coefficients to a separate plot
    xloc=[]; val=[]; err=[]; chi=[]; chi_err=[];
    for n in range(corrStart,v['pdf'].numHTs[0]+corrStart):
        xloc.append(n+0.05*int(k))
        val.append(v['pdf'].pAvg['C^{t4}_%d'%n])
        err.append(np.sqrt(v['pdf'].pCov[('C^{t4}_%d'%n,'C^{t4}_%d'%n)]))
        # chi.append(v['pdf'].pAvg['chi2'])
        # chi_err.append(np.sqrt(v['pdf'].pCov[('chi2','chi2')]))
        # ax_t4_chi.errorbar(k,v['pdf'].pAvg['chi2'],yerr=np.sqrt(v['pdf'].pCov[('chi2','chi2')]),\
        #                    capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')


        print(k)
        print(v['pdf'].pAvg['chi2'])
        print(np.sqrt(v['pdf'].pCov[('chi2','chi2')]))
        ax_twist_chi.errorbar(k,v['pdf'].pAvg['chi2'],yerr=np.sqrt(v['pdf'].pCov[('chi2','chi2')]),\
                           capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')
        
    ax_t4.errorbar(xloc,val,yerr=err,capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')
    # ax_t4_chi.errorbar(n,v['pdf'].pAvg['chi2'],yerr=np.sqrt(v['pdf'].pCov[('chi2','chi2')]),\
    #                    capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')
    print("XLOC = ")
    print(xloc)
                       
    
    xloc=[]; val=[]; err=[]; chi=[]; chi_err=[];
    for n in range(corrStart,v['pdf'].numHTs[1]+corrStart):
        # xloc.append(n+0.025*int(k))
        # xloc.append(n+1+0.05*int(k)) # hacky to make ax_t4
        xloc.append(n+0.025*int(k))
        val.append(v['pdf'].pAvg['C^{t6}_%d'%n])
        err.append(np.sqrt(v['pdf'].pCov[('C^{t6}_%d'%n,'C^{t6}_%d'%n)]))
        # chi.append(v['pdf'].pAvg['chi2'])
        # chi_err.append(np.sqrt(v['pdf'].pCov[('chi2','chi2')]))
        # ax_t6_chi.errorbar(k,v['pdf'].pAvg['chi2'],yerr=np.sqrt(v['pdf'].pCov[('chi2','chi2')]),\
        #                    capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')
    ax_t6.errorbar(xloc,val,yerr=err,capsize=3.0,color=mainColors[int(k)],label=lab,fmt='o')
    # ax_t6_chi.errorbar(xloc,chi,yerr=chi_err,capsize=3.0,color=mainColors[int(k)],\
    #                    label=lab,fmt='o')


# sys.exit()

#####################
# SET RANGES
#####################
# ax.set_ylim([0.5,1.5])
ax.set_ylim([0.7,1.3])
ax.set_xlim([0,1])
ax.hlines(1,-2,2,hcolor,ls='--')
ax.set_ylabel(r'$%s_{q_%s/N}\left(x\right)/\langle %s_{q_%s/N}^{\rm ref}\left(x\right)\rangle$'%\
              (pdf,pdfType,pdf,pdfType))
# ax.set_ylabel(r'$%s_{q_%s/N}\left(x\right)/\left\langle %s_{q_%s/N}^{\\rm ref}\left(x;[%s,%s;%s,%s]\right)\right\rangle$'%\
#               (pdf,pdfType,pdf,pdfType,options.normPZ_ranges.split('_')[0].split('.')[0],options.normPZ_ranges.split('_')[0].split('.')[1],\
#                options.normPZ_ranges.split('_')[1].split('.')[0],options.normPZ_ranges.split('_')[1].split('.')[1]))
ax.set_xlabel(r'$x$')


#################################
# ADD TEXT TO ANY AXES
#################################
rangex=ax.get_xlim()[1]-ax.get_xlim()[0]
rangey=ax.get_ylim()[1]-ax.get_ylim()[0]
print(rangey)

# #####
# # WITHOUT/WITH MATELEM SYSTEMATIC
# #####
# ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.3*rangey,\
#         s=r'$%s_{q_%s/N}^{\rm ref}\left(x\right):\ {\rm no\ matrix\ element\ systematic}$'\
#         %(pdf,pdfType),fontsize=16,color='dimgray')
# ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.26*rangey,\
#         s=r'$%s_{q_%s/N}\left(x\right):\ {\rm with\ matrix\ element\ systematic}$'\
#         %(pdf,pdfType),fontsize=16,color=mainColors[0])

# #####
# # VARIABLE PRIOR WIDTHS
# #####
# ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.36*rangey,\
#         s=r'$%s_{q_%s/N}^{\rm ref}\left(x\right):\ {\rm Default\ prior\ widths}$'\
#         %(pdf,pdfType),fontsize=16,color='dimgray')
# ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.32*rangey,\
#         s=r'$%s_{q_%s/N}\left(x\right):\ {\rm Wide\ prior\ widths}$'\
#         %(pdf,pdfType),fontsize=16,color=mainColors[0])
# ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.28*rangey,\
#         s=r'$%s_{q_%s/N}\left(x\right):\ {\rm Thin\ prior\ widths}$'\
#         %(pdf,pdfType),fontsize=16,color=mainColors[1])
# ax.text(ax.get_xlim()[0]+0.1*rangex,ax.get_ylim()[0]+0.9*rangey,\
#         s=r'${\tt With\ matrix\ element\ systematic}$',
#         fontsize=17,color='black')

#####
# P\IN[1,6] Z\IN[2,8] PDF COMPARED TO P\IN[1,6] Z\IN[1,8] & P\IN[2,6] Z\IN[2,8]
#               (WITH MATELEM SYSTEMATIC)
#####
ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.36*rangey,\
        s=r'$%s_{q_%s/N}^{\rm ref}\left(x\right):\ {\rm Default\ prior\ widths}$'\
        %(pdf,pdfType),fontsize=16,color='dimgray')
ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.32*rangey,\
        s=r'$%s_{q_%s/N}\left(x\right):\ {\rm Default\ prior\ widths}$'\
        %(pdf,pdfType),fontsize=16,color=mainColors[0])
ax.text(ax.get_xlim()[0]+0.2*rangex,ax.get_ylim()[0]+0.28*rangey,\
        s=r'$%s_{q_%s/N}\left(x\right):\ {\rm Default\ prior\ widths}$'\
        %(pdf,pdfType),fontsize=16,color=mainColors[1])
ax.text(ax.get_xlim()[0]+0.1*rangex,ax.get_ylim()[0]+0.9*rangey,\
        s=r'${\tt With\ matrix\ element\ systematic}$',
        fontsize=17,color='black')






##########################################
# Make the legend entries
##########################################
handles, labels = ax.get_legend_handles_labels()
dumH=[]
dumL=[]
numLabels=len(labels)
halfNumLabels=int(numLabels/2)

for l in range(0,halfNumLabels):
    dumH.append( (handles[l],handles[halfNumLabels+l]) )
    dumL.append( labels[l] )
customHandles=dumH
customLabels=tuple(dumL)

ax.legend(customHandles,customLabels,framealpha=LegendFrameAlpha,loc='lower left',\
          title=r'$\left(N_{lt},N_{az},N_{t4},N_{t6}\right)\quad [p^{min}_{\rm latt},p^{max}_{\rm latt};z^{min}/a,z^{max}/a]$',ncol=1)


    

# for a in [ax_t4, ax_t6]:
#     a.legend(framealpha=FrameAlpha,loc='upper right',\
#              title=r'$[p_{min},p_{max};z_{min},z_{max}]$')


if options.logx:
    ax.semilogx()
######################################
# ROTATE TICK LABELS ON CHI2 SUBPLOTS
######################################
# for a in [ax_t4_chi, ax_t6_chi]:
#     plt.setp(a.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
plt.setp(ax_twist_chi.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

######################
# SAVE THE FIGURES
######################
pmins=[]; pmaxs=[]; zmins=[]; zmaxs=[];
for a in options.altPZ_ranges.split('/'):
    c=a.replace('_','.').split('.')
    pmins.append(int(c[0])); pmaxs.append(int(c[1]))
    zmins.append(int(c[2])); zmaxs.append(int(c[3]))
        
pminRange="%s-%s"%(min(pmins),max(pmins))
pmaxRange="%s-%s"%(min(pmaxs),max(pmaxs))
zminRange="%s-%s"%(min(zmins),max(zmins))
zmaxRange="%s-%s"%(min(zmaxs),max(zmaxs))
        
fig.savefig("jacobi-pdf_compare.jacobi-truncs_%s.cutranges_pmin%s_pmax%s_zmin%s_zmax%s%s.%s"\
            %(options.jacobiCorrections,pminRange,pmaxRange,zminRange,zmaxRange,suffix,form),\
            dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
# twist4.savefig("coeff_twist-4_variations.jacobi-truncs_%s.cutranges_pmin%s_pmax%s_zmin%s_zmax%s%s.%s"\
#                %(options.jacobiCorrections,pminRange,pmaxRange,zminRange,zmaxRange,suffix,form),\
#                dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
# twist6.savefig("coeff_twist-6_variations.jacobi-truncs_%s.cutranges_pmin%s_pmax%s_zmin%s_zmax%s%s.%s"\
#                %(options.jacobiCorrections,pminRange,pmaxRange,zminRange,zmaxRange,suffix,form),\
#                dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)


twist.savefig("coeff_twist-4and6_variations.jacobi-truncs_%s.cutranges_pmin%s_pmax%s_zmin%s_zmax%s%s.%s"\
              %(options.jacobiCorrections,pminRange,pmaxRange,zminRange,zmaxRange,suffix,form),\
              dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)

    

plt.show()
