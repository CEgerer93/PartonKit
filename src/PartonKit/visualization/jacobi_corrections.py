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


parser.add_option("-C", "--corrections", type="str", default='x.x.x',
                  help='Disc.Twist4.Twist6 corrections (default = "x.x.x")')
parser.add_option("-f", "--correctionFile", type="str", default='',
                  help='correctionFile (default = '')')
parser.add_option("-L", "--fullFitFile", type="str", default='',
                  help="For absolute error of correction terms, grab the full fit file to access leading twist contribution (default='')")
parser.add_option("--allJacobiOrders", type="str", default='x.x.x.x',
                  help="Truncation orders of full jacobi polynomial fit to pitd - Only relevant if fullFitFile given above (default='x.x.x.x')")
parser.add_option("-r", "--reality", type="str", default='',
                  help='Re -or- Im (default = "")')
parser.add_option("-c", "--cfgs", type="int", default=0,
                  help='Gauge configs (default = "")')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-t", "--cuts", type="str", default='x.x.x.x',
                  help='Cuts on p/z <pmin>.<pmax>.<zmin>.<zmax> (default = "x.x.x.x")')
parser.add_option("-z", "--zmax", type="int", default=-1,
                  help='Max zsep to show correction for (default = -1)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("--htLtRelErr", action="store_true", default=False, dest="htLtRelErr",
                  help='Plot relative error of HT corrections to LT (default = False)')
parser.add_option("--alphaS", type="float", default=0.303,
                  help="AlphaS of fits (default = 0.303)")


# Parse the input arguments
(options, args) = parser.parse_args()
gauge_configs=options.cfgs
comp=options.reality
PDF.alphaS = options.alphaS # SET ALPHAS in pdf_utils
DIRAC=options.insertion

zCutMin, zCutMax = tuple(options.cuts.split('.')[2:])
zCutMin=int(zCutMin)
zCutMax=int(zCutMax)

corrStart=-1
if comp == 'Re':
    corrStart=1
if comp == 'Im':
    corrStart=0

amplitude='M' if DIRAC == 8 else 'Y'

#############
# Take fullFitFile name to help make saved figure names
#############
saveFigPrefix=options.fullFitFile.replace(".txt","")

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
if options.lightBkgd:
    mainColors=['red','purple','blue','green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']
else:
    mainColors=['red','purple',cm.twilight(40),'green','white','orange','magenta',(0.1,1,0.1),'black','gray','gray']

dispColors = []
if options.lightBkgd == 1:
    for dc in [ cm.ocean(d) for d in np.linspace(0,0.85,options.zmax+1) ]:
        dispColors.append(dc)

if options.lightBkgd == 0:
    for dc in [ cm.RdYlGn_r(d) for d in np.linspace(0,1,options.zmax+1) ]:
        dispColors.append(dc)
    
    # # for dc in [ cm.Spectral(d) for d in np.linspace(0,1,options.zmax+1) ]:
    # for dc in [ cm.Spectral(d) for d in np.linspace(1,0,options.zmax+1) ]:
    #     dispColors.append(dc)
    




# A figure to show jacobi determined corrections
fig=plt.figure(figsize=(12,10))
ax=fig.gca()
if options.fullFitFile != "":
    fig_relErr=plt.figure(figsize=(12,10))
    ax_relErr=fig_relErr.gca()

# ax.set_xlabel(r'$\nu$')
# ax.set_xlim([0,15])
# # ax.set_xticks(np.linspace(0,20,11))
# ax.set_xticks(np.linspace(0,14,8))

axes=[ax];
if options.fullFitFile != "":
    axes.append(ax_relErr)
for a in axes:
    a.set_xlabel(r'$\nu$')
    a.set_xlim([0,15])
    a.set_xticks(np.linspace(0,14,8))



lab=''
if int(options.corrections.split('.')[0]) != 0:
    lab+='az,'
if int(options.corrections.split('.')[1]) != 0:
    lab+='t4,'
if int(options.corrections.split('.')[2]) != 0:
    lab+='t6'
if lab[-1] == ',':
    lab=lab[:-1]


truncStart=None

############################
############################
########## YLIMS ###########
############################
############################

##### DEF PRIORS/NO MATELEM SYSTEMATIC #####
# ax.set_ylim([-0.025,0.005]) # qv disc !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([-0.04,0.006]) # qv all-ht !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([-0.05,0.01]) # qv disc & all-ht !!!! p\in[2,6] z\in[2,8]
##### DEF PRIORS/WITH MATELEM SYSTEMATIC #####
# ax.set_ylim([-0.0002,0.022]) # qv disc !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([-0.03,0.011]) # qv all-ht !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([-0.05,0.01]) # qv disc & all-ht !!!! p\in[2,6] z\in[2,8]


##### DEF PRIORS/NO MATELEM SYSTEMATIC #####
# ax.set_ylim([-0.005,0.041]) # q+ all-ht !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([0.0,0.2]) # q+ disc !!!! EITHER p\in[1,6] z\in[2,8] or p\in[2,6] z\in[2,8]
# ax.set_ylim([-0.005,0.07]) # q+ all-ht !!!! p\in[2,6] z\in[2,8]
##### DEF PRIORS/WITH MATELEM SYSTEMATIC #####
# ax.set_ylim([-0.02,0.025]) # q+ all-ht !!!! p\in[1,6] z\in[2,8]
# ax.set_ylim([0.0,0.16]) # q+ disc !!!! EITHER p\in[1,6] z\in[2,8] or p\in[2,6] z\in[2,8]
ax.set_ylim([-0.005,0.04]) # q+ all-ht !!!! p\in[2,6] z\in[2,8]

####################################################################################
####################################################################################
####################################################################################


ax.set_ylabel(r'$\mathfrak{%s}\ \mathfrak{%s}^{%s}\left(\nu,z^2\right)$'%(comp,amplitude,lab))
if options.fullFitFile != "":
    ax_relErr.set_ylabel(r'$\mathfrak{%s}\ \mathfrak{%s}^{%s}\left(\nu,z^2\right)/\langle\mathfrak{%s}\ \mathfrak{%s}^{lt}\left(\nu,z^2\right)\rangle$'%(comp,amplitude,lab,comp,amplitude))
    
if comp == 'Re':
    truncStart=1
if comp == 'Im':
    truncStart=0

##################################################################################################
# Optionally read in the full jacobi fit parameter file - for relative error of corrections below
##################################################################################################
if options.fullFitFile != "":
    numLT, numAZ, numT4, numT6, numT8, numT10 = [ int(i) for i in options.allJacobiOrders.split('.') ]

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
    with open(options.fullFitFile) as ptr:
        for cnt, line in enumerate(ptr):
            L=line.split(' ')
            for n, k in enumerate(paramOrder):
                fitParams[k].append(float(L[n]))


    fullJacobiFit=PDF.pitdJacobi('entire', mainColors,\
                                 {k: fitParams[k] for k,v in fitParams.items()},\
                                 [numLT,numAZ,numT4,numT6], True, gauge_configs, comp,\
                                 corrStart, dirac=DIRAC)
    fullJacobiFit.parse()
###################################################################################################


# Reset fit parameter dictionaries
fitParams={'\\alpha': [], '\\beta': []}
paramOrder=['\\alpha', '\\beta']
corrTags=['az', 't4', 't6']
for i, c in enumerate(options.corrections.split('.')):
    for n in range(truncStart,int(c)+truncStart):
        fitParams.update({'C^{%s}_%d'%(corrTags[i],n): []})
        paramOrder.append('C^{%s}_%d'%(corrTags[i],n))

        
# Access the jacobi fit parameters
with open(options.correctionFile) as ptr:
    for cnt, line in enumerate(ptr):
        # Capture line and remove spaces
        L=line.split(' ')
        for n, k in enumerate(paramOrder):
            fitParams[k].append(float(L[n]))


truncOrders=[0]
[truncOrders.append(int(k)) for k in options.corrections.split('.')]
print(truncOrders)
jac=PDF.pitdJacobi('corr', mainColors, fitParams,\
                   truncOrders,True,gauge_configs,comp,corrStart,dirac=DIRAC)
jac.parse()
jac.printFitParams()


###################################################################################
# PLOT ABSOLUTE ERROR OF CORRECTIONS (I.E. CORRECTIONS ALONE W/O NORMALIZATION
###################################################################################
for z in range(options.zmax,0,-1):
    jac.plotPITD(ax,z,dispColors[z-1] if (z>=zCutMin) and (z<=zCutMax) else 'gray')
#----------------------------------------------------------------------------------


#####################################################################
# PLOT RELATIVE ERROR OF CORRECTIONS TO LEADING-TWIST CONTRIBUTION
#####################################################################
if options.fullFitFile != "" and options.htLtRelErr:
    for z in range(options.zmax,0,-1):
        relErr_Avg=[]; relErr_Err=[]
        for n, nu in enumerate(jac.nu):
            relErr_Avg.append(jac.pitd(nu,z)/fullJacobiFit.lt_pitd(nu,z))
            relErr_Err.append(jac.pitdError(nu,z)/fullJacobiFit.lt_pitd(nu,z))
        
        ax_relErr.plot(jac.nu,relErr_Avg,color=dispColors[z-1])
        ax_relErr.fill_between(jac.nu,[a+e for a,e in zip(relErr_Avg,relErr_Err)],\
                               [a-e for a,e in zip(relErr_Avg,relErr_Err)],\
                               color=dispColors[z-1] if (z>=zCutMin) and (z<=zCutMax) else 'gray',\
                               alpha=0.3)
#    ax_relErr.set_ylim([-3,3])



# Plot some dummy bands to facilitate an easy legend
dum=np.linspace(-5,-4,100)
# figAxPairs=[(fig, ax, 'lower left')] # qv
figAxPairs=[(fig, ax, 'upper left')] # q+
if options.fullFitFile != "":
    figAxPairs.append((fig_relErr, ax_relErr, 'upper left'))
for f,a,Loc in figAxPairs:
    bands={}
    bandCollect=[]
    bandLabels=[]
    for z in range(1,options.zmax+1):
        bands.update({z: {'avg': None, 'err': None}})
        bands[z]['avg'], = a.plot(dum,100*np.ones(100),\
                                  color=dispColors[z-1] if (z>=zCutMin) and (z<=zCutMax) else 'gray')
        bands[z]['err'] = a.fill_between(dum,100+z,100-z,\
                                         color=dispColors[z-1] if (z>=zCutMin) and (z<=zCutMax) else 'gray',\
                                         alpha=0.35,label=r'$z=%d$'%z)
    
        bandCollect.append( (bands[z]['avg'], bands[z]['err']) )
        bandLabels.append(r'$z=%d$'%z)

    # Add the legend
    a.legend(bandCollect,bandLabels,loc=Loc,fontsize=20,framealpha=FrameAlpha)



yrange=ax.get_ylim()[1]-ax.get_ylim()[0]
ymin=ax.get_ylim()[0]

for n,c in enumerate(['p^{min}_{\\rm latt}','p^{max}_{\\rm latt}','z^{min}/a','z^{max}/a']):
    # qv
    if comp == 'Re':
        # ax.text(11,ymin+yrange*(0.3-n*0.05),r'$%s=%d$'%(c,int(options.cuts.split('.')[n])),fontsize=20) # ------> w/o matelem systematic
        ax.text(11,ymin+yrange*(0.9-n*0.05),r'$%s=%d$'%(c,int(options.cuts.split('.')[n])),fontsize=20) # ------> w/ matelem systematic
    # q+
    if comp == 'Im':
        ax.text(11,ymin+yrange*(0.9-n*0.05),r'$%s=%d$'%(c,int(options.cuts.split('.')[n])),fontsize=20)


fig.savefig("jac-corrections-trunc_%d%d%d.%s%s.%s"%\
            (int(options.corrections.split('.')[0]),int(options.corrections.split('.')[1]),\
             int(options.corrections.split('.')[2]),saveFigPrefix,\
             suffix,form),\
            dpi=600,transparent=truthTransparent,bbox_inches='tight',format=form)



if options.fullFitFile != "":
    fig_relErr.savefig("jac-corrections-trunc_relErr_%d%d%d.%s%s.%s"%\
                       (int(options.corrections.split('.')[0]),int(options.corrections.split('.')[1]),\
                        int(options.corrections.split('.')[2]),saveFigPrefix,\
                        suffix,form),\
                       dpi=600,transparent=truthTransparent,bbox_inches='tight',format=form)




fig_itdCorr=plt.figure(figsize=(12,10))
ax_itdCorr=fig_itdCorr.gca()
ax_itdCorr.set_xlabel(r'$\nu$')
ax_itdCorr.set_ylabel(r'$\mathfrak{%s}\ \mathfrak{%s}\left(\nu,z^2\right)$'%(comp,amplitude))

bands={}
bandCollect=[]
bandLabels=[]
for zi in range(8, 1, -3):
    fullJacobiFit.plotPITD(ax_itdCorr,zi,dispColors[zi-1])
    jac.plotPITD(ax_itdCorr,zi,dispColors[zi-1],hatch='x')

    bands.update({zi: {'avg': None, 'err': None}})
    bands[zi]['avg'], = ax_itdCorr.plot(dum,0.005*np.ones(100),color=dispColors[zi-1])
    bands[zi]['err'] = ax_itdCorr.fill_between(dum,0.005+zi,0.005-zi,color=dispColors[zi-1],\
                                               alpha=0.35,label=r'$z=%d$'%zi)
    
    bandCollect.append( (bands[zi]['avg'], bands[zi]['err']) )
    bandLabels.append(r'$z=%d$'%zi)

    # Add the legend
    ax_itdCorr.legend(bandCollect,bandLabels,loc='upper right',fontsize=20,framealpha=FrameAlpha)

ax_itdCorr.set_xlim([0,15])
ax_itdCorr.set_ylim([-0.1,1.05])
ax_itdCorr.set_xticks(np.linspace(0,14,8))

#####################################################################
# Add text to ax_itdCorr plot labeling leading-twist and corrections
#####################################################################
rangex=ax_itdCorr.get_xlim()[1]-ax_itdCorr.get_xlim()[0]
rangey=ax_itdCorr.get_ylim()[1]-ax_itdCorr.get_ylim()[0]

textFontSize=24
if comp == 'Re':
    ax_itdCorr.text(0.5*rangex,0.5*rangey,\
                    s=r'$\mathfrak{%s}\ \mathfrak{%s}_{lt}\left(\nu,z^2\right)$'%(comp,amplitude),\
                    fontsize=textFontSize,color='dimgray')
    ax_itdCorr.text(0.1*rangex,0.15*rangey,\
                    s=r'$\mathfrak{%s}\ \mathfrak{%s}_{%s}\left(\nu,z^2\right)$'%(comp,amplitude,lab),\
                    fontsize=textFontSize,color='dimgray')
    # Add arrows from leading-twist/corrections annotations to some curves
    ax_itdCorr.annotate("", xy=(0.47*rangex, 0.3*rangey), xytext=(0.56*rangex, 0.48*rangey),\
                        c='gray',arrowprops=dict(arrowstyle="->"))
    ax_itdCorr.annotate("", xy=(0.25*rangex, 0.02*rangey), xytext=(0.15*rangex, 0.13*rangey),\
                        color='gray',arrowprops=dict(arrowstyle="->"))
if comp == 'Im':
    ax_itdCorr.text(0.6*rangex,0.6*rangey,\
                    s=r'$\mathfrak{%s}\ \mathfrak{%s}_{lt}\left(\nu,z^2\right)$'%(comp,amplitude),\
                    fontsize=textFontSize,color='dimgray')
    ax_itdCorr.text(0.4*rangex,0.15*rangey,\
                    s=r'$\mathfrak{%s}\ \mathfrak{%s}_{%s}\left(\nu,z^2\right)$'%(comp,amplitude,lab),\
                    fontsize=textFontSize,color='dimgray')
    # Add arrows from leading-twist/corrections annotations to some curves
    ax_itdCorr.annotate("", xy=(0.54*rangex, 0.46*rangey), xytext=(0.66*rangex, 0.58*rangey),\
                        c='gray',arrowprops=dict(arrowstyle="->"))
    ax_itdCorr.annotate("", xy=(0.37*rangex, 0.06*rangey), xytext=(0.45*rangex, 0.13*rangey),\
                        color='gray',arrowprops=dict(arrowstyle="->"))

#####################################################################
#####################################################################
#####################################################################


fig_itdCorr.savefig("jac-corrections-lt-trunc-compare_%d%d%d.%s%s.%s"%\
                    (int(options.corrections.split('.')[0]),int(options.corrections.split('.')[1]),\
                     int(options.corrections.split('.')[2]),saveFigPrefix,
                     suffix,form),\
                    dpi=600,transparent=truthTransparent,bbox_inches='tight',format=form)
# fig_itdCorr.savefig("lt-jac-corrections-trunc-compare_%d%d%d.%s%s.pmin%d_pmax%d_zmin%d_zmax%d.%s"%\
#                     (int(options.corrections.split('.')[0]),int(options.corrections.split('.')[1]),\
#                      int(options.corrections.split('.')[2]),options.reality,suffix,\
#                      int(options.cuts.split('.')[0]),int(options.cuts.split('.')[1]),\
#                      int(options.cuts.split('.')[2]),int(options.cuts.split('.')[3]),form),\
#                     dpi=600,transparent=truthTransparent,bbox_inches='tight',format=form)

    
plt.show()


