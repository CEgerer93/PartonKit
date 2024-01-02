#!/usr/bin/python3

#####/dist/anaconda/bin/python

#####
# PLOT JACK DATA STORED IN AN H5 FILE
#####

import numpy as np
import h5py
import matplotlib.font_manager as fm
from matplotlib import cm
import pylab # to save figures to file
import sys,optparse
from collections import OrderedDict
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
sys.path.append('/home/colin/QCD/pseudoDists/strFuncViz')
from pitd_util import *
import fit_util
import gpd_utils
from common_fig import *

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-H", "--h5File", type="string", default="",
                  help='H5 file(s) to plot <x>:<x> (default = "x:x")')
parser.add_option("-d", "--dtypeName", type="string", default="",
                  help='Datatype name(s) to access <d>:<d> (default = "d:d")')
parser.add_option("--fit2pt",type="str", default='',
                  help='H5 file containing 2pt fits (default = '')')
parser.add_option("-s", "--singleFig", type="int", default=0,
                  help='If more than one h5 file, default to multiple panels or plot within one (default = 0)')
parser.add_option("-z", "--zRange", type="string", default="",
                  help='Min/Max zsep in h5 <zmin>.<zmax> (default = '')')
parser.add_option("-p", "--boostMoms", type="str", default='',
                  help='Fin/Ini momenta of rpGITD numerator <pf/pi>=X.X.X/X.X.X (default = '')')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-a","--aLatt", type="float", default=0.0,
                  help='Lattice spacing (default = 0.0)')
parser.add_option("-A", "--axesOverride", type="int", default=0,
                  help='Override vertical axes range in evo/match plots (default = 0)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("--showFig",action="store_true",default=False,dest="showFig")
parser.add_option("-D", action="store_false", default=True, dest="useDispRelnForMomTrans",
                  help='Do not use dispersion relation in computing momentum transfer (default = True)')
parser.add_option("--dblRatio", action="store_true", default=False, dest="dblRatio",
                  help='Use double ratio in forming pGITD')

insertions = { 3: { 'Redstar': 'b_b1xDA__J1_T1pP', 'CK': None },
               8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK': 'insertion_gt' },
               11: { 'Redstar': 'a_a1xDA__J1_T1pM', 'CK': None} }



# Parse the input arguments
(options, args) = parser.parse_args()
cfgs=options.cfgs
zmin, zmax=tuple(int(z) for z in options.zRange.split('.'))
pf, pi=tuple(p for p in options.boostMoms.split('/'))



pgitdData=gpd_utils.h5Plot(options.h5File,options.dtypeName,options.fit2pt,cfgs,pf,pi,zmin,zmax,options.Lx)
pgitdData.initAxes()
pgitdData.read2pts()
pgitdData.showPGITDData()



##############################################################################
# INCLUDE TEXT FOR MOMENTUM/DISP & MODIFY AXES + INCLUDE XI/T TEXT FOR PGITDS
##############################################################################
for K,V in pgitdData.ampDict.items():
    for c, a in V.items():
        ax=a['ax']
        yrange=ax.get_ylim()[1]-ax.get_ylim()[0]
        xtent=ax.get_xlim()[1]-ax.get_xlim()[0]
        
        fooBuff=0
        # fooBuff=-0.6 # for vertical shifting of text, until some automatic implementation
        if pf != pi: # formerly 0.56*xtent below
            ax.text(ax.get_xlim()[0]+0.7*xtent,(0.92+fooBuff)*yrange+ax.get_ylim()[0],\
                    r'$\vec{p}_f=\left(%s,%s,%s\right)$'%\
                    tuple(p for p in pf.split('.')),fontsize=18)
            ax.text(ax.get_xlim()[0]+0.7*xtent,(0.87+fooBuff)*yrange+ax.get_ylim()[0],\
                    r'$\vec{p}_i=\left(%s,%s,%s\right)$'%\
                    tuple(p for p in pi.split('.')),fontsize=18)
            ax.text(ax.get_xlim()[0]+0.7*xtent,(0.8+fooBuff)*yrange+ax.get_ylim()[0],\
                    r'$\xi=%.4f$'%skewness(pf,pi),fontsize=18)
            ax.text(ax.get_xlim()[0]+0.7*xtent,(0.75+fooBuff)*yrange+ax.get_ylim()[0],\
                    r'$t=%.4f\thinspace{\rm GeV}^2$'%\
                    momTrans(pgitdData.Ef,pf,pgitdData.Ei,pi,pgitdData.m,options.Lx,options.aLatt,options.useDispRelnForMomTrans),\
                    fontsize=18)
            
            
            # Extend xrange so legend will fit nicely
            ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]+3*pgitdData.STEP])
        else:
            plt.text(3.5,0.4*yrange+ax.get_ylim()[0],r'$\vec{p}=\left(%s,%s,%s\right)$'%\
                     tuple(p for p in options.boostMoms.split('/').split('.')),fontsize=18)
#------------------------------------------------------------------------------------




# Fetch all the unique labels on each fig, and make the legends
for amp,comps in pgitdData.ampDict.items():
    for fig, ax in [tuple([figAx['fig'],figAx['ax']]) for comp,figAx in comps.items()]:
        # Get the handles/labels
        handles, labels = ax.get_legend_handles_labels()
        Ncol=1
        
        # Catch indices of first unique displacement
        catch=[labels.index("%s"%z) for z in np.unique(labels)]
        
        # labelsNew=[]; handlesNew=[]
        labelsNew=[labels[n] for n in catch]
        handlesNew=[handles[n] for n in catch]
        
        
        hlDict={}
        for n in range(0,len(labelsNew)):
            hlDict.update({labelsNew[n]: handlesNew[n]})
            
        # Sort the legend keys on displacement
        labelsNewSort=sorted(labelsNew,key=lambda s: int(s.replace('$z=','').replace('$','')))
        
        
        handlesNewSort=[]
        for k in labelsNewSort:
            handlesNewSort.append(hlDict[k])
            
            
        # Turn errorbars in legend to be horizontal
        for h in handlesNew:
            h.has_xerr=True; h.has_yerr=False
            
        by_label=OrderedDict(zip(labelsNewSort,handlesNewSort))
        ax.legend(by_label.values(),by_label.keys(),framealpha=LegendFrameAlpha,ncol=Ncol,loc=1)
    


# Include string for denoting if double ratio was used
dblRatioStr='-dblRatio' if options.dblRatio else ''
for K,V in pgitdData.ampDict.items():
    for c,f in V.items():
        f['fig'].savefig("rpGITD%s_%s_g%i_pf%s_pi%s_%s%s.%s"%\
                         (dblRatioStr,K,options.insertion,toStr(pf),toStr(pi),c,suffix,form),\
                         dpi=400,transparent=truthTransparent,\
                         bbox_inches='tight',format=form)


if options.showFig:
    plt.show()
