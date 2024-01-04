#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pylab # to save figures to file
import optparse
from util.ratio_util import correlator

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-x", "--complexity", type="int", default=0,
                  help='Component of Npt function to analyze [Real=1 Imag=2] (default = 1)')
parser.add_option("-i", "--pi", type="string", default="0.0.0",
                  help='Source momentum <pix>.<piy>.<piz> (default = 0.0.0)')
parser.add_option("-q", "--Q", type="string", default="0.0.0",
                  help='Momentum transfer <qx>.<qy>.<qz> (default = 0.0.0)')
parser.add_option("-f", "--pf", type="string", default="0.0.0",
                  help='Sink momentum <pfx>.<pfy>.<pfz> (default = 0.0.0)')
parser.add_option("-z", "--vecZ", type="string", default="0.0.0",
                  help='Displacement vector <x>.<y>.<z> (default = 0.0.0)')
parser.add_option("-N", "--nptCorrsFile", type="string", default="",
                  help='List file containing full path/name for Npt correlator(s) (default="")')
parser.add_option("-r", "--twoptSrcCorrsFile", type="string", default="",
                  help='File containing full path/name for Src 2pt correlator (default="")')
parser.add_option("-k", "--twoptSnkCorrsFile", type="string", default="",
                  help='File containing full path/name for Snk 2pt correlator (default="")')
parser.add_option("-g", "--chromaGamma", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-o", "--output", type="string", default="",
                  help='Override name of plateau plot\n (default = "pgitd_plateau_pf<snkX><snkY><snkZ>_pi<srcX><srcY><srcZ>_rows<snkRow><srcRow>_g<gamma>")')
parser.add_option("-w", "--srcRow", type="string", default="X",
                  help='Source interpolator row (default = X)')
parser.add_option("-y", "--snkRow", type="string", default="X",
                  help='Sink interpolator row (default = X)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("-s", "--showFig", type="int", default=0,
		  help='Show the figure (default = 0)')


# Parse the input arguments
(options, args) = parser.parse_args()

################
# INITIALIZE GLOBAL PROPERTIES OF FIGURES
################
# Finalize the figures
plt.rc('text', usetex=True)
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rc('xtick.major',size=5)
plt.rc('ytick.major',size=5)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)
truthTransparent=False
FrameAlpha=1
# Optionally swap default black labels for white
if options.lightBkgd == 0:
    truthTransparent=True
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['legend.frameon' ] = False
    plt.rc('axes',edgecolor='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('text',color='white')
    FrameAlpha=0


cfgs=options.cfgs
complexity=options.complexity
gamma=options.chromaGamma
srcMom=options.pi
snkMom=options.pf
insMom=options.Q
vecZ=options.vecZ

redstarFactor=np.sqrt(2)

# Grab and read the Src 2pt correlator
with open(options.twoptSrcCorrsFile) as pt:
    twoptSrcCorr=pt.read().rstrip()
corrSrc2pt=np.loadtxt(twoptSrcCorr, delimiter=' ', skiprows=1)
# Grab and read the Snk 2pt correlator
with open(options.twoptSnkCorrsFile) as pt:
    twoptSnkCorr=pt.read().rstrip()
corrSnk2pt=np.loadtxt(twoptSnkCorr, delimiter=' ', skiprows=1)

#############################################
#       Grab and read Npt correlators
#############################################
with open(options.nptCorrsFile) as tpt:
    nptCorrs=tpt.read().splitlines()
data3pt=[]
for F in nptCorrs:

    # If we are dealing with more than one file, parse and combine before proceeding
    if len(F.split(' ')) == 1:
        data3pt.append(np.loadtxt(F, delimiter=' ', skiprows=1))
        continue
    elif len(F.split(' ')) == 2:
        dumCorr1=np.loadtxt(F.split(' ')[0], delimiter=' ', skiprows=1)
        dumCorr2=np.loadtxt(F.split(' ')[1], delimiter=' ', skiprows=1)
        print dumCorr1
        mergedCorr=mergeCorrs(dumCorr1,dumCorr2,gamma)
        data3pt.append(mergedCorr)
    else:
        raise Exception("Don't know how to handle %i correlators"%len(F.split(' ')))


src2pt=correlator(cfgs,len(corrSrc2pt)/cfgs,corrSrc2pt,1,1.0)
snk2pt=correlator(cfgs,len(corrSnk2pt)/cfgs,corrSnk2pt,1,1.0)
src2pt.makeJks()
snk2pt.makeJks()
src2pt.avgJks()
snk2pt.avgJks()



R=[]
for n, c in enumerate(data3pt):
    dum3pt=correlator(cfgs,len(c)/cfgs,c,complexity,redstarFactor)
    dum3pt.makeJks()
    dum3pt.avgJks()
    # Make the ratio for this data3pt entry
    r=ratio(dum3pt,src2pt,snk2pt)
    r.avgratio()
    r.avgratioErr()
    # Append
    R.append(r)




# fig=plt.figure()
ax=plt.gca()
# title_str = "Nucleon effective "+charge+r" Charge - Isoclover $32^3\times64$    $m_\pi=%d$ MeV    $a=%.3f$ fm"%(356,0.098)+"\n"+title_str
# plt.title(title_str)

if complexity == 1:
    ax.set_ylabel(r'Re $M_%s\left(p_f,p_i,z_3\right)$'%gammaDict[gamma],fontsize=20)
if complexity == 2:
    ax.set_ylabel(r'Im $M_%s\left(p_f,p_i,z_3\right)$'%gammaDict[gamma],fontsize=20)
ax.set_xlabel(r'$\left(\tau-T/2\right)a^{-1}$',fontsize=20)

def makeTimeArrays(N):
    time=np.zeros(N)
    for i in range(0,N):
        time[i]=i-N/2

    return time


for n, r in enumerate(R):
    ts=makeTimeArrays(len(r.Avg))
    # ts=[i-len(r.Avg)/2 for i in range(0,len(r.Avg))]
    # print ts
    r.plotRatio(ts,ax)

# Set a suitable vertical range
ymin, ymax = ax.get_ylim()
Ymin = ymin - 0.8*(ymax-ymin)
Ymax = ymax + 0.8*(ymax-ymin)
ax.set_ylim([Ymin,Ymax])

# Add the momenta and displacements to plot
xext=abs(ax.get_xlim()[1]-ax.get_xlim()[0])
yext=abs(ax.get_ylim()[1]-ax.get_ylim()[0])

ax.text(ax.get_xlim()[0]+0.125*xext,ax.get_ylim()[0]+0.1*yext,\
        r'$\vec{p}_i=\left(%s,%s,%s\right)\quad\quad\vec{p}_f=\left(%s,%s,%s\right)\quad\quad\vec{z}=\left(%s,%s,%s\right)$'\
        %(srcMom.split('.')[0],srcMom.split('.')[1],srcMom.split('.')[2],\
          snkMom.split('.')[0],snkMom.split('.')[1],snkMom.split('.')[2],\
          vecZ.split('.')[0],vecZ.split('.')[1],vecZ.split('.')[2]),fontsize=18)
ax.text(ax.get_xlim()[0]+0.125*xext,ax.get_ylim()[0]+0.05*yext,\
        r'$r_i=\mu_{%s}\quad\quad\quad\quad r_f=\mu_{%s}$'%(options.srcRow,options.snkRow),fontsize=18)


ax.legend(fontsize=16,ncol=3,loc=9,markerscale=1,fancybox=True)

# Name the output
if options.output == "":
    output_name="ppdf_plateau_pf%s%s%s_pi%s%s%s_r%s%s_g%i_z%s%s%s_c%i"\
        %(snkMom.split('.')[0],snkMom.split('.')[1],snkMom.split('.')[2],\
          srcMom.split('.')[0],srcMom.split('.')[1],srcMom.split('.')[2],\
          options.snkRow,options.srcRow,gamma,\
          vecZ.split('.')[0],vecZ.split('.')[1],vecZ.split('.')[2],complexity)
else:
    output_name=options.output
plt.savefig(output_name,transparent=truthTransparent,bbox_inches='tight',pad_inches=0)#,dpi=500)
if options.showFig == 1:
    plt.show()
