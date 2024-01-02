#!/usr/bin/python3.8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
import sys,optparse
from collections import OrderedDict
import pdf_utils as PDF

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-d", "--dglap", type="str", default="",
                  help='Plot effect of DGLAP kernel on pITD data (default = '')')
parser.add_option("-m", "--matching", type="str", default="",
                  help='Plot effect of MATCHING kernel on pITD data (default = '')')
parser.add_option("-a", "--amplitude", type="str", default="",
                  help='Amplitude (e.g. M,N,Y...) whose evolution/matching is being visualized (default = '')')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-Z", "--h5ZCut", type="str", default="x.x",
                  help='Cut on <zmin>.<zmax> in any h5 file (default = x.x)')
parser.add_option("-P", "--h5PCut", type="str", default="x.x",
                  help='Cut on <pmin>.<pmax> in any h5 file (default = x.x)')
parser.add_option("--lightBkgd", action="store_true", default=False, dest="lightBkgd",
                  help='Format figs for light or dark background (default = False)')
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")

# Parse the input arguments
(options, args) = parser.parse_args()

gauge_configs=options.cfgs
DIRAC=options.insertion
PDF.alphaS = options.alphaS # SET ALPHAS in pdf_utils

dispColors = ['gray']
# for dc in [ cm.gnuplot(d) for d in np.linspace(0,1,int(options.h5ZCut.split('.')[1])+1) ]:
for dc in [ cm.turbo(d) for d in np.linspace(0,1,int(options.h5ZCut.split('.')[1])+1) ]:
    dispColors.append(dc)

###############################################################################################
# INITIALIZE GLOBAL PROPERTIES OF FIGURES
###############################################################################################
# Finalize the figures
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams.update({'errorbar.capsize': 2})
plt.rc('xtick.major',size=10) #5 - for 6342 jacobi truncation fits
plt.rc('ytick.major',size=10) #5 - for 6342 jacobi truncation fits
plt.rc('xtick',labelsize=25)  #20 - for 6342 jacobi truncation fits
plt.rc('ytick',labelsize=25)  #20 - for 6342 jacobi truncation fits
plt.rc('axes',labelsize=30)   #20 - for 6342 jacobi truncation fits
truthTransparent=False
FrameAlpha=0.8
legendFaceColor="white"
suffix=''
form='pdf'
# Optionally swap default black labels for white
if not options.lightBkgd:
    truthTransparent=True
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rc('axes',edgecolor='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('text',color='white')
    # plt.rcParams['legend.columnspacing'] = 1.25
    FrameAlpha=0.0
    # legendFaceColor="#1b212c"
    suffix='.dark'
    form='png'
#------------------------------------------------------------------------------------------------


if options.dglap != "":
    fig_evo_real=plt.figure(figsize=(12,10)); ax_evo_real=fig_evo_real.gca()
    fig_evo_imag=plt.figure(figsize=(12,10)); ax_evo_imag=fig_evo_imag.gca()
    for a in [ax_evo_real,ax_evo_imag]:
        a.set_xlim([0,10])
        a.set_xlabel(r'$\nu$')
    # ax_evo_real.set_ylim([-0.5,5])
    # ax_evo_imag.set_ylim([-5,5])
    ax_evo_real.set_ylabel(r'$B\ \otimes\ \mathfrak{Re}\ \mathfrak{%s}$'%options.amplitude)
    ax_evo_imag.set_ylabel(r'$B\ \otimes\ \mathfrak{Im}\ \mathfrak{%s}$'%options.amplitude)

    PDF.h5Plot(options.dglap,options.h5ZCut,options.h5PCut,[ax_evo_real],\
               [ax_evo_imag],options.cfgs,"evoKernel",dispColors,dirac=DIRAC,\
               oldH5Hierarchy=False)
    
            # # Plot this info
            # thisColor=dispColors[abs(zsep)]
            # if comp == 0:
            #     ax_evo_real.errorbar(ioffeTime,matelem,yerr=error,fmt='o',mfc="None",mec=thisColor,\
            #                          color=thisColor,capsize=3,label=r'$z=%s$'%abs(zsep))
            # if comp == 1:
            #     ax_evo_imag.errorbar(ioffeTime,matelem,yerr=error,fmt='^',mfc="None",mec=thisColor,\
            #                          color=thisColor,capsize=3,label=r'$z=%s$'%abs(zsep))

# Open MATCH data and plot contents
if options.matching != "":
    fig_match_real=plt.figure(figsize=(12,10)); ax_match_real=fig_match_real.gca()
    fig_match_imag=plt.figure(figsize=(12,10)); ax_match_imag=fig_match_imag.gca()
    for a in [ax_match_real, ax_match_imag]:
        a.set_xlim([0,10])
        a.set_xlabel(r'$\nu$')
    # ax_match_real.set_ylim([-5,0.5])
    # ax_match_imag.set_ylim([-5,5])
    ax_match_real.set_ylabel(r'$L\ \otimes\ \mathfrak{Re}\ \mathfrak{%s}$'%options.amplitude)
    ax_match_imag.set_ylabel(r'$L\ \otimes\ \mathfrak{Im}\ \mathfrak{%s}$'%options.amplitude)

    PDF.h5Plot(options.matching,options.h5ZCut,options.h5PCut,[ax_match_real],\
               [ax_match_imag],options.cfgs,"matchingKernel",dispColors,dirac=DIRAC,\
               oldH5Hierarchy=False)
    
            # # Plot this info
            # thisColor=dispColors[abs(zsep)]
            # if comp == 0:
            #     ax_match_real.errorbar(ioffeTime,matelem,yerr=error,fmt='o',mfc="None",\
            #                            mec=thisColor,color=thisColor,capsize=3,\
            #                            label=r'$z=%s$'%abs(zsep))
            # if comp == 1:
            #     ax_match_imag.errorbar(ioffeTime,matelem,yerr=error,fmt='^',mfc="None",\
            #                            mec=thisColor,color=thisColor,capsize=3,\
            #                            label=r'$z=%s$'%abs(zsep))


#########
# Make the legends for each axes
#########
for fig, ax, loc in [(fig_evo_real,ax_evo_real,'upper left'), (fig_evo_imag,ax_evo_imag, 'upper left'),\
                     (fig_match_real,ax_match_real, 'upper right'), (fig_match_imag,ax_match_imag, 'upper right')]:
    try:
        # Get the handles/labels
        handles, labels = fig.axes[0].get_legend_handles_labels()
        Ncol=2

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
                           framealpha=FrameAlpha,ncol=Ncol,loc=loc)# ,bbox_to_anchor=(0.9,1))
        for t in legend.get_texts():
            t.set_ha('left') # ha is alias for horizontalalignment
            t.set_position((shift,0))
        frame = legend.get_frame()
        frame.set_facecolor(legendFaceColor)
    except:
        continue

###########
# Save the figures
###########
fig_evo_real.savefig(options.dglap.replace("h5","Re_as%.4f%s.%s"%(options.alphaS,suffix,form)),\
                     dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
fig_evo_imag.savefig(options.dglap.replace("h5","Im_as%.4f%s.%s"%(options.alphaS,suffix,form)),\
                     dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
fig_match_real.savefig(options.matching.replace("h5","Re_as%.4f%s.%s"%(options.alphaS,suffix,form)),\
                       dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
fig_match_imag.savefig(options.matching.replace("h5","Im_as%.4f%s.%s"%(options.alphaS,suffix,form)),\
                       dpi=400,transparent=truthTransparent,bbox_inches='tight',pad_inches=0.1,format=form)
        
plt.show()
