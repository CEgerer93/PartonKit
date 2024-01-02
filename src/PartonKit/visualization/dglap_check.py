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

import pdf_utils as PDF

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-d", "--dataType", type="str", default='',
                  help='Data to read - either <pitd> or <itd> (default = '')')
parser.add_option("-r", "--realResults", type="str", default='',
                  help='Real fits dglap check (default = '')')
parser.add_option("-i", "--imagResults", type="str", default='',
                  help='Imag fits dglap check (default = '')')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-H", "--H5", type="str", default="",
                  help='H5 file of <pitd> or <itd> data that was fit (default = "")')
parser.add_option("-Z", "--h5ZCut", type="str", default="x.x",
                  help='Cut on <zmin>.<zmax> in any h5 file (default = x.x)')
parser.add_option("-P", "--h5PCut", type="str", default="x.x",
                  help='Cut on <pmin>.<pmax> in any h5 file (default = x.x)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')



# Parse the input arguments
(options, args) = parser.parse_args()

gauge_configs=options.cfgs
H5=options.H5

################
# INITIALIZE GLOBAL PROPERTIES OF FIGURES
################
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
plt.rc('xtick.major',size=10)
plt.rc('ytick.major',size=10)
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
plt.rc('axes',labelsize=30)
truthTransparent=False
FrameAlpha=0.8
legendFaceColor="white"
suffix=''
form='pdf'
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
    # legendFaceColor="#1b212c"
    suffix='.dark'
    form='png'


# Color according the z value
mainColors=None
# mainColors=['blue','green','red','white','purple','orange','magenta',(0.1,1,0.1),'black','gray','gray']
# mainColors=['blue','green','red','purple','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']
if options.lightBkgd:
    mainColors=['red','purple','blue','green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']
else:
    mainColors=['red','purple',cm.twilight(40),'green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']

dispColors = ['gray']
if options.lightBkgd == 1:
    for dc in [ cm.brg(d) for d in np.linspace(0,1,17) ]:
        dispColors.append(dc)

if options.lightBkgd == 0:
    for dc in [ cm.RdYlGn_r(d) for d in np.linspace(0,1,17) ]:
        dispColors.append(dc)





fig_pitd_perz, ax_pitd_perz=plt.subplots(4,4,figsize=(18,14))

# fig_pitd_perz.subplots_adjust(top=0.965,bottom=0.140,left=0.102,\
#                               right=0.979,hspace=0.295,wspace=0.200)

fig_pitd_perz.add_subplot(111, frameon=False)
ax_pitd_perz_all=fig_pitd_perz.gca()
for a in [ax_pitd_perz_all]:
    a.tick_params(labelcolor='none', top=False, bottom=False,\
                  left=False, right=False)
    a.set_xlabel(r'$\nu$')
    
if options.realResults != '':
    ax_pitd_perz_all.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{M}(\nu,z^2)$')
    if options.dataType == 'itd':
        ax_pitd_perz_all.set_ylabel(r'$\mathfrak{Re}\ Q(\nu,\mu^2)$')
if options.imagResults != '':
    ax_pitd_perz_all.set_ylabel(r'$\mathfrak{Im}\ \mathfrak{M}(\nu,z^2)$')
    if options.dataType == 'itd':
        ax_pitd_perz_all.set_ylabel(r'$\mathfrak{Im}\ Q(\nu,\mu^2)$')


# A figure to show how alpha changes w/ zsep
fig_alpha_vs_z=plt.figure(figsize=(12,10))
ax_alpha_vs_z=fig_alpha_vs_z.gca()
ax_alpha_vs_z.set_xlabel(r'$z/a$')
ax_alpha_vs_z.set_ylabel(r'$\alpha\left(z/a\right)$')
# ax_alpha_vs_z.set_xlim([0,17])
# ax_alpha_vs_z.set_ylim([-0.2,4.2])

# ax_alpha_vs_z.set_xlim([0,14])
# ax_alpha_vs_z.set_ylim([-0.2,1.4])

ax_alpha_vs_z.set_xlim([0,16.5])
ax_alpha_vs_z.set_xticks(np.linspace(0,16,9))

### FOR THE REAL ALPHA vs. Z
ax_alpha_vs_z.set_ylim([-0.4,1.4])
ax_alpha_vs_z.set_yticks(np.linspace(-0.4,1.4,10))

### FOR THE IMAG ALPHA vs. Z
# ax_alpha_vs_z.set_ylim([-2.75,5.25])
# ax_alpha_vs_z.set_yticks(np.linspace(-2,5,8))


    
perZAxes=ax_pitd_perz.flat




if options.realResults != "":

    fig_pitd_perz.subplots_adjust(top=0.972,bottom=0.140,left=0.102,\
                                  right=0.982,hspace=0.244,wspace=0.249)

    perZRanges=[0.93,0.8,0.65,0.4,0.2,0,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-1.0,-1.0,-1.0,-1.0]
    for z, ax in enumerate(ax_pitd_perz.flat):
        if z == 0:
            ax.set_xlim(0,1.5)
        else:
            ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
        ax.set_ylim(perZRanges[z],1)
        
        if z > 11:
            ax.set_ylim(perZRanges[z],2)

        try:
            PDF.h5Plot(options.H5,z+1,options.h5PCut,ax,\
                       None,None,None,gauge_configs,options.dataType,dispColors)
        except:
            continue
    
    for z, File in enumerate(options.realResults.split(':')):

        # Reset the fit parameter dictionaries
        fitParamsAll2V={'chi2': [], 'alpha': [], 'beta': []}
        params2=['chi2', 'alpha', 'beta']

        # Read fit parameters from this param2File
        with open(File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                # As of 02-23-2021 Fit params include norm, which isnt needed in qval, so remove it
                del L[1]
                for n, k in enumerate(params2):
                    fitParamsAll2V[k].append(float(L[n]))
                    
                    
        jam=PDF.pdf2('v','C',dispColors[z+1],
                     {'chi2': fitParamsAll2V['chi2'], 'alpha': fitParamsAll2V['alpha'],\
                      'beta': fitParamsAll2V['beta'],'norm': None},bool(1),\
                     gauge_configs)
        
        jam.parse()
        jam.printFitParams()
        jam.plotITDReal(perZAxes[z])
        
        xmin, xmax = perZAxes[z].get_xlim()
        ymin, ymax = perZAxes[z].get_ylim()
        
        rangeX=abs(xmax-xmin)
        rangeY=abs(ymax-ymin)
        
        
        # Add the chi2 to the subplot
        perZAxes[z].text(xmin+0.05*rangeX,ymin+0.05*rangeY,r'$\chi^2_r = %.4f(%.4f)$'%
                         (jam.pAvg['chi2'],np.sqrt(jam.pCov[('chi2','chi2')])),fontsize=16)
        
        
        
        # Lastly add the alpha + error to ax_alpha_vs_z axes
        if z < 17:
            ax_alpha_vs_z.errorbar(z+1,jam.pAvg['alpha'],yerr=np.sqrt(jam.pCov[('alpha','alpha')]),\
                                   fmt='o',mfc=None,mec=dispColors[z+1],color=dispColors[z+1],\
                                   capsize=3,label=r'$z=%s$'%str(z+1))

if options.imagResults != "":

    fig_pitd_perz.subplots_adjust(top=0.972,bottom=0.140,left=0.102,\
                                  right=0.982,hspace=0.452,wspace=0.249)

    perZRanges=[0.25,0.5,0.6,0.7,0.7,0.75,0.75,0.75,0.85,0.85,1.0,1.1,1.3,1.4,1.4,1.4]
    for z, ax in enumerate(ax_pitd_perz.flat):
        ax.set_xlim(0, (2*np.pi/32)*(z+1.75)*float(options.h5PCut.split('.')[1]) )
        ax.set_ylim(0,perZRanges[z])
        if z > 11:
            ax.set_ylim(-1.5,perZRanges[z])
        PDF.h5Plot(options.H5,z+1,options.h5PCut,None,\
                   None,ax,None,gauge_configs,options.dataType,dispColors)
    
    for z, File in enumerate(options.imagResults.split(':')):

        # Reset the fit parameter dictionaries
        fitParamsAll2P={'chi2': [], 'normP': [], 'alphaP': [], 'betaP': []}
        params2=['chi2', 'normP', 'alphaP', 'betaP']

        # Read fit parameters from this param2File
        with open(File) as ptr:
            for cnt, line in enumerate(ptr):
                L=line.split(' ')
                for n, k in enumerate(params2):
                    fitParamsAll2P[k].append(float(L[n]))
                    
                    
        jam=PDF.pdf2('+','C',dispColors[z+1],
                     {'chi2': fitParamsAll2P['chi2'], 'alpha': fitParamsAll2P['alphaP'],\
                      'beta': fitParamsAll2P['betaP'],'norm': fitParamsAll2P['normP']},\
                     bool(1),gauge_configs)
        
        jam.parse()
        jam.printFitParams()
        jam.plotITDImag(perZAxes[z])
        
        xmin, xmax = perZAxes[z].get_xlim()
        ymin, ymax = perZAxes[z].get_ylim()
        
        rangeX=abs(xmax-xmin)
        rangeY=abs(ymax-ymin)
        
        
        # Add the chi2 to the subplot
        perZAxes[z].text(xmin+0.15*rangeX,ymin+0.1*rangeY,r'$\chi^2_r = %.4f(%.4f)$'%
                         (jam.pAvg['chi2'],np.sqrt(jam.pCov[('chi2','chi2')])),fontsize=13)
        
        
        
        # Lastly add the alpha + error to ax_alpha_vs_z axes
        if z < 17:
            ax_alpha_vs_z.errorbar(z+1,jam.pAvg['alpha'],yerr=np.sqrt(jam.pCov[('alpha','alpha')]),\
                                   fmt='o',mfc=None,mec=dispColors[z+1],color=dispColors[z+1],\
                                   capsize=3,label=r'$z=%s$'%str(z+1))



ax_alpha_vs_z.legend(ncol=2,framealpha=FrameAlpha,loc='upper left')


fig_pitd_perz.savefig("dglap_checks%s.%s"%(suffix,form),dpi=600,transparent=truthTransparent,\
                      bbox_inches='tight',pad_inches=0.1,format=form)
fig_alpha_vs_z.savefig("dglap_checks_alpha_vs_z%s.%s"%(suffix,form),dpi=600,\
                       transparent=truthTransparent,\
                       bbox_inches='tight',pad_inches=0.1,format=form)

plt.show()
