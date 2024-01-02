#!/usr/bin/python3
#######################################################
# READ IN PSEUDO-ITD DATA AND SEE IF DGLAP IS PRESENT #
#######################################################
import numpy as np
import matplotlib.pyplot as plt
import sys,optparse
import pylab
import scipy.special as spec
from scipy import optimize as opt

sys.path.append("/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo")
import fit_util
import corr_util
import pitd_util


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-r","--datSR",type="str", default='',
                  help='H5 file containing summed ratio data (default = '')')
parser.add_option("-f","--fitSR",type="str", default='',
                  help='H5 file containing summed ratio fit results (default = '')')
parser.add_option("--cfgs", type="int", default=0,
                  help='Ensemble configs (default = 0)')
parser.add_option("-t","--fitTSeries", type="str", default='x.x:x.x',
                  help='tmin.step.tmax of fit (default = x.x.x:x.x.x)')
parser.add_option("--pf", type="str", default='x.x.x',
                  help='Final Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--pi", type="str", default='x.x.x',
                  help='Initial Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--rows", type="str", default='x.x',
                  help='Snk/src rows <snk row>.<src row> (default = x.x)')
parser.add_option("-z","--zsep", type="str", default='x.x.x',
                  help='Displacement in format <dx>.<dy>.<dz>; "null" to ignore displacement (default = x.x.x)')
parser.add_option("-g","--gamma", type="int", default=0,
                  help='Chroma gamma (default = 0)')
parser.add_option("-c","--component", type="str", default='real',
                  help='Read component (default = real)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("--HR", action="store_true", default=False, dest="HR",
                  help='Save fig in high resolution')
parser.add_option("-s", action="store_true", default=False, dest="showFig",
                  help='Put figure to screen')
parser.add_option("--pdfH5", action="store_true", default=False, dest="pdfH5",
                  help='Read fit results from h5 file that does not include snk/src row indices - i.e. from a pITD analysis (default = false)')

# Parse the input arguments
(options, args) = parser.parse_args()


#####################
# Parse options
#####################
pf=pitd_util.toStr(options.pf)
pi=pitd_util.toStr(options.pi)
rowf, rowi=tuple([r for r in options.rows.split('.')])

z=pitd_util.toStr(options.zsep)
if z == 'null':
    z=None

compStr=''
if options.component == 'real':
    compStr='Re'
elif options.component == 'imag':
    compStr='Im'


################
# Whether distinct pf,pi are needed in labels
################
momLabel=None
if pf != pi:
    momLabel="\\vec{p}_f,\\vec{p}_i,\\vec{z};T"
else:
    momLabel="\\vec{p},\\vec{z};T"

################
# Gamma dicts
################
gammaDict={1: r'$\Gamma=\gamma_x$', 2: r'$\Gamma=\gamma_y$', 8: r'$\Gamma=\gamma_4$'}
#------------------------------------------------------------------


######################################
# INITIALIZE FIGURE
######################################
plt.rc('text', usetex=True)
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
plt.rc('xtick.major',size=5)
plt.rc('ytick.major',size=5)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)
axLabelSize=18
truthTransparent=False
FrameAlpha=0.8
if options.lightBkgd == 0:
    truthTransparent=True
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rc('axes',edgecolor='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('text',color='white')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    FrameAlpha=0
    
fig=plt.figure(figsize=(8,6))
ax=fig.gca()
ax.set_xlabel(r'$T/a$',fontsize=axLabelSize)
ax.set_ylabel(r'$\mathfrak{%s}\ R\left(%s\right)$'%(compStr,momLabel),fontsize=axLabelSize)
ax.set_xlim([3,15])
############################################################################################



tseries={'data': {'min': 4, 'step': 2, 'max': 14, 'numT': 6},
         'fit': {'min': 0, 'step': 0, 'max': 0}}

bandColors=['darkblue','brown','forestgreen','saddlebrown','orange']


bands={} # Dictionary for error bands
for n,f in reversed(list(enumerate(options.fitTSeries.split(':')))):
    bands.update({n: {'avg': None, 'err': None}})
    
    tseries['fit']['min']=int(f.split('.')[0])
    tseries['fit']['step']=int(f.split('.')[1])
    tseries['fit']['max']=int(f.split('.')[2])

    # Construct a summed ratio fit instance
    fitSR=fit_util.SR(options.fitSR,options.cfgs,"pf%s"%pf,"pi%s"%pi,rowf,rowi,"zsep%s"%z,options.gamma,options.component,\
                      tseries['fit']['min'],tseries['fit']['max'],pdfH5=options.pdfH5)
    fitSR.parse()
    fitSR.printFitParams()

    # The time slices at which to sample the fit
    tFit=np.linspace(tseries['fit']['min']-0.5*tseries['fit']['step'],
                     tseries['fit']['max']+0.5*tseries['fit']['step'],500)
    theFit={'avg': [fitSR.func(T) for T in tFit],
            'err': [fitSR.err(T) for T in tFit]}


    bands[n]['avg'], = plt.plot(tFit,theFit['avg'],color=bandColors[n])
    bands[n]['err'] = plt.fill_between(tFit,[x-y for x,y in zip(theFit['avg'],theFit['err'])],
                                       [x+y for x,y in zip(theFit['avg'],theFit['err'])],
                                       color=bandColors[n],alpha=0.5,
                                       label=r'${\rm Fit}\ T/a\in\left[%s,%s\right]$'\
                                       %(tseries['fit']['min'],tseries['fit']['max']),lw=0)


################################
# Add data from h5 file to plot
################################
datSR=corr_util.corrSR(options.datSR,options.cfgs,tseries['data']['numT'],\
                       pf,pi,rowf,rowi,options.component,z,options.gamma)
datSR.readDat()
datSR.makeAvg()
datSR.jk()
datSR.avgJks()
datSR.makeCov()
tDat=np.linspace(tseries['data']['min'],tseries['data']['max'],
                 int((tseries['data']['max']-tseries['data']['min'])/tseries['data']['step'])+1)

datHandle=None
for n, t in enumerate(tDat):
    datHandle=plt.errorbar(t, datSR.avg[n], np.sqrt(datSR.cov[n][n]),fmt='o',color='orangered',\
                           capsize=3,zorder=3,alpha=0.65)
#------------------------------------------------------------------------------------------------


# ax.set_ylim([-0.91,23.5]) # pz=5, z=0,2 real
# ax.set_ylim([-0.28,4.2]) # pz=5, z=5, real
# ax.set_ylim([-0.7,0.15]) # pz=5, z=8, real
# ax.set_ylim([-0.91,4.5]) # pz=5, z=0, imag
# ax.set_ylim([0.56,12.75]) # pz=5, z=2, imag
# ax.set_ylim([0.56,6.4]) # pz=5, z=5, imag
# ax.set_ylim([-0.12,1.45]) # pz=5, z=8, imag
# ax.set_ylim([-6,8])

############################################
# Add text indicating momenta/displacements
############################################
yrange=ax.get_ylim()[1]-ax.get_ylim()[0]
if pf != pi:
    plt.text(3.5,0.54*yrange+ax.get_ylim()[0],gammaDict[options.gamma],fontsize=18)
    plt.text(3.5,0.47*yrange+ax.get_ylim()[0],r'$\vec{p}_f=\left(%s,%s,%s\right)$'%\
             tuple(p for p in options.pf.split('.')),fontsize=18)
    plt.text(3.5,0.4*yrange+ax.get_ylim()[0],r'$\vec{p}_i=\left(%s,%s,%s\right)$'%\
             tuple(p for p in options.pi.split('.')),fontsize=18)
else:
    plt.text(3.5,0.4*yrange+ax.get_ylim()[0],r'$\vec{p}=\left(%s,%s,%s\right)$'%\
             tuple(p for p in options.pi.split('.')),fontsize=18)
if z is not None:
    plt.text(3.5,0.33*yrange+ax.get_ylim()[0],r'$\vec{z}=\left(%s,%s,%s\right)$'%\
             tuple(z for z in options.zsep.split('.')),fontsize=18)

#     plt.text(3.5,0.8*yrange+ax.get_ylim()[0],r'$\vec{p}=\left(%s,%s,%s\right)$'%\
#              tuple(p for p in options.pi.split('.')),fontsize=18)
# if z is not None:
#     plt.text(3.5,0.73*yrange+ax.get_ylim()[0],r'$\vec{z}=\left(%s,%s,%s\right)$'%\
#              tuple(z for z in options.zsep.split('.')),fontsize=18)

#####################
# Manage the legend
#####################
legendBands=[]; legendLabels=[]
for n,t in enumerate(options.fitTSeries.split(':')):
    legendBands.append( (bands[n]['avg'], bands[n]['err']) )
    legendLabels.append( r'{\rm Fit} $T/a\in\left[%s,%s\right]$'%(t.split('.')[0], t.split('.')[2]) )
legendBands.append( datHandle )
legendLabels.append( r'$R\left(%s\right)$'%momLabel )
plt.legend(legendBands, legendLabels, fontsize=18, framealpha=FrameAlpha)

# plt.legend([(bands[0]['avg'], bands[0]['err']), (bands[1]['avg'], bands[1]['err']),\
#             (bands[2]['avg'], bands[2]['err']), (datHandle)],\
#            [r'{\rm Fit} $T/a\in\left[4,14\right]$', r'{\rm Fit} $T/a\in\left[6,14\right]$',\
#             r'{\rm Fit} $T/a\in\left[8,14\right]$', r'$R\left(%s\right)$'%momLabel],\
#            fontsize=18,framealpha=FrameAlpha)
#---------------------------------------------------------------------------------------------


suffix=''
form='pdf'
dpi=10
if options.lightBkgd == 0:
    suffix='.dark'
    form='png'
if options.HR:
    dpi=400

plt.savefig("SR-Fit_g%i_pf%s_rf%s_pi%s_ri%s_zsep%s_%s%s.%s"%\
            (options.gamma,pf,rowf,pi,rowi,z,compStr,suffix,form),\
            dpi=dpi,transparent=truthTransparent,bbox_inches='tight',format=form)
if options.showFig:
    plt.show()
