#!/usr/bin/python3
#######################################################
# READ IN PSEUDO-ITD DATA AND SEE IF DGLAP IS PRESENT #
#######################################################
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys,optparse
import pylab
import scipy.special as spec
from scipy import optimize as opt
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
import pitd_util
import fit_util
import corr_util
from common_fig import *

    

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-f","--corrH5",type="str", default='',
                  help='H5 file containing correlator data (default = '')')
parser.add_option("-F","--corrFitH5",type="str", default='',
                  help='H5 file containing fits to correlator data (default = '')')
parser.add_option("-g", "--cfgs", type="int", default=0,
                  help='Ensem configs (default = 0)')
# parser.add_option("-t","--tSeries", type="str", default='x.x.x:x.x.x',
#                   help='tmin.step.max of data(sets) (default = x.x.x:x.x.x)')
# parser.add_option("-T","--tfitSeries", type="str", default='x.x.x:x.x.x',
#                   help='tmin.step.max of fit(s) (default = x.x.x:x.x.x)')
parser.add_option("--pf", type="str", default='x.x.x',
                  help='Final Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--pi", type="str", default='x.x.x',
                  help='Initial Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("-z","--dispTag", type="str", default='z+20',
                  help='Displacement tag in format <0> or <z+disp> (default = z+20)')
parser.add_option("-c","--component", type="str", default='',
                  help='Read component (default = )')
parser.add_option("-N", "--StoN", type="float", default=2.0,
                  help='Signal to noise cut-off (default = 2.0)')
parser.add_option("-S", "--showStoN", type="int", default=0,
                  help='Include subplot illustrating S/N ratios vs. T (default = 0)')
parser.add_option("--gpd",action="store_true", default=False, dest="gpd",
                  help='In accessing 2pts, use operators from GPD analysis')
parser.add_option("--showFig",action="store_true", default=False, dest="showFig",
                  help='Whether to put figure to screen')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')

# Parse the input arguments
(options, args) = parser.parse_args()
cfgs=options.cfgs




fig=plt.figure(figsize=(7,9.5))
ax=fig.gca()
ax.set_xlabel(r'$T/a$') # ,fontsize=axLabelSize)
# ax.set_ylabel(r'$C_2\left(\vec{p},T\right)$',fontsize=axLabelSize)
ax.set_ylabel(r'$aE_{\rm eff}\left(\vec{p},T\right)$') # ,fontsize=axLabelSize)
ax.set_xlim([1,19])
ax.set_ylim([0.4,1.5])




# bandColors=['darkblue','brown','forestgreen','saddlebrown','orange']
bandColors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'cyan', 'yellow']

h5Corr=h5py.File(options.corrH5,'r')
h5Fit=h5py.File(options.corrFitH5,'r')


tSeries={}
# Get all the tSeries of the data
for p,v in h5Corr.items():
    for k, op in v['t0_avg'].items():
        tmax = len(op['rows_11/real/data'][0])
        tSeries.update({p: "%s.1.%s"%(0,tmax-1)})

tfitSeries={}
# Get all the tfitSeries
for k,v in h5Fit['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real'].items():
    tmin, tmax = list(v.keys())[0].replace('tfit_','').split('-')
    tfitSeries.update({k: '%s.1.%s'%(tmin,tmax)})



#################################
#      Run over the fits
#################################
bands={} # dictionary for error bands
for n,p in enumerate(options.pf.split(':')): # Take --pf to delineate unique momenta to loop over
    # Get momenta for this fit
    pf=pitd_util.toStr(p)
    pi=pitd_util.toStr(options.pi.split(':')[n])
    
    bands.update({n: {'avg': None, 'err': None}})
    corrFit=fit_util.twoState(h5Fit,cfgs,pf,pi,options.component,tfitSeries["p%s"%pf],\
                              col=bandColors[n],z=None,gam=None)
    corrFit.parse()
    corrFit.plot(ax,eff=True,fitTMin=corrFit.tmin,fitTMax=corrFit.tmax)


    ################################
    # Add data from h5 file to plot
    ################################
    corr=corr_util.corr2pt(h5Corr,cfgs,tSeries["p%s"%pi],pi,\
                           options.component,bandColors[n],options.StoN,gpdOps=options.gpd)
    corr.readDat()
    corr.makeAvg()
    corr.jk()
    corr.avgJks()
    corr.makeCov()
    corr.plot(ax,eff=True)


####################
# Manage the legend
####################
handles, labels = ax.get_legend_handles_labels()
customHandles=[]
customLabels=[]

for n,l in enumerate(labels):
    if l not in customLabels:
        handles[n].has_xerr=True; handles[n].has_yerr=False
        customLabels.append(l)
        customHandles.append(handles[n])

ax.legend(customHandles,customLabels,\
          loc='upper right', ncol=2,handletextpad=0.1,columnspacing=0.1,\
          framealpha=LegendFrameAlpha,fontsize=11)


suffix=''
form='pdf'
if options.lightBkgd == 0:
    suffix='.dark'
    form='png'

plt.savefig("eff_corr-view%s.%s"%(suffix,form),\
            dpi=400,transparent=truthTransparent,bbox_inches='tight',format=form)


if options.showFig:
    plt.show()
