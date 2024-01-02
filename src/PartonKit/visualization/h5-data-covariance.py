#!/usr/bin/python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pylab
import sys,optparse

from pdf_utils import insertions


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);
parser.add_option("-d", "--h5File", type="str", default='',
                  help='H5 file to view data covariance (default = '')')
parser.add_option("-g", "--cfgs", type="int", default=1,
                  help='Gauge configs (default = 1)')
parser.add_option("-z", "--h5ZCut", type="str", default="x.x",
                  help='Cut on <zmin>.<zmax> in any h5 file (default = x.x)')
parser.add_option("-p", "--h5PCut", type="str", default="x.x",
                  help='Cut on <pmin>.<pmax> in any h5 file (default = x.x)')
parser.add_option("-I", "--insertion", type="int", default="8",
                  help='Gamma matrix to consider (default = 8)')
parser.add_option("-c", "--comp", type="str", default='',
                  help='Re -or- Im (default = '')')
parser.add_option("-t", "--dtypeName", type="str", default='',
                  help='H5 datatype name (default = '')')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("--oldH5Hierarchy", action="store_true", default=False, dest="h5Hierarchy",
                  help='Read associated h5s in old format (b4 zsep/pf_pi changes)')

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
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rc('xtick.major',size=10)
plt.rc('ytick.major',size=10)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('axes',labelsize=32)
truthTransparent=False
FrameAlpha=0.7
legendFaceColor="white"
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
    FrameAlpha=1.0
    legendFaceColor="#1b212c"

    
fig=plt.figure(figsize=(12,10))
# fig_im=plt.figure(figsize=(12,10))
ax=fig.gca()
# ax.xtick_params(pad=24)
# ax_im=fig_im.gca()


h5In = h5py.File(options.h5File,'r')
cfgs=options.cfgs
zmin=int(options.h5ZCut.split('.')[0]); zmax = int(options.h5ZCut.split('.')[1]);
pmin=int(options.h5PCut.split('.')[0]); pmax = int(options.h5PCut.split('.')[1]);
Z = [zi for zi in range(zmin,zmax+1)]
P = [pi for pi in range(pmin,pmax+1)]



# for comp in ["Re", "Im"]:

avgMatDict={}
matDict={}


for m in range(pmin,pmax+1):
    ptag="pz%d"%m
    if not(options.h5Hierarchy):
        ptag="pf00%d_pi00%d"%(m,m)
        
    for z in Z:
        ztag="zsep%d"%z
        if not(options.h5Hierarchy):
            ztag="zsep00%d"%z

        # thisPZKey=("z/a=%d"%z,"p=%d"%m)
        thisPZKey=("p=%d"%m,"z/a=%d"%z)
        # Make an empty array for this z/m key in matDict
        matDict.update({thisPZKey: []})

        avgMat = 0.0
        for g in range(0,cfgs):
            ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                  (insertions[options.insertion]['Redstar'],\
                                   ztag,ptag,options.comp,options.dtypeName)][g]
            avgMat += mat
            matDict[thisPZKey].append(mat)

        # print(matDict[thisPZKey])

        avgMat *= (1.0/cfgs)
        avgMatDict.update({thisPZKey: avgMat})
        # print(avgMatDict[thisPZKey])
        # sys.exit()



#####
# NOW BUILD THE DATA COVARIANCE HEATMAP
#####
dataCovView=np.zeros((len(avgMatDict),len(avgMatDict)))
for ni, ki in enumerate(avgMatDict.keys()):
    for nj, kj in enumerate(avgMatDict.keys()):
        key=(ki,kj)
        dum=0.0
        for g in range(0,cfgs):
            dum+=(matDict[ki][g] - avgMatDict[ki])*\
                  (matDict[kj][g] - avgMatDict[kj])
        dum*=((1.0*(cfgs-1))/cfgs)

        dataCovView[ni,nj]=dum

# Normalize dataCovView to diagonal entries
diag=[dataCovView[d,d] for d in range(0,len(dataCovView))]
for i in range(0,len(dataCovView)):
    for j in range(0,len(dataCovView)):
        dataCovView[i,j]/=np.sqrt(diag[i]*diag[j])
        # print(dataCovView[i,j])



dax=ax.matshow(dataCovView,cmap='cividis') #'viridis')# bwr')
cbar = fig.colorbar(dax)
cbar.set_ticks(np.arange(-1.0,1.1,0.5)) # [-1. , -0.5,  0. ,  0.5,  1. ]
cbar.set_ticklabels(['${}$'.format(tkval) for tkval in [-1, -0.5, 0, 0.5, 1]])
                     # [1, 2, 3, 4, 5]]
#                     np.arange(-1.0,1.1,0.5)) # [-1. , -0.5,  0. ,  0.5,  1. ]

# cbar.ax.set_xticklabels(['${}$'.format(tkval) for tkval in [1, 2, 3, 4, 5]])

ax.set_xticks(np.arange(len(avgMatDict)),minor=False)
ax.set_yticks(np.arange(len(avgMatDict)))
# ax.set_yticklabels('$[%s,%s]$'%(k[0],k[1]) for k in avgMatDict.keys())
# ax.set_xticklabels('$[%s,%s]$'%(k[0],k[1]) for k in avgMatDict.keys())

# ax.xaxis.set_major_locator(MultipleLocator(len(Z)))

ax.set_yticklabels('$%s$'%k[0] for k in avgMatDict.keys())
ax.set_xticklabels('$%s$'%k[0] for k in avgMatDict.keys())


# For tidiness, only keep one momentum label per set of zseps
[l.set_visible(False) for (i,l) in enumerate(ax.get_xticklabels()) if i % len(Z) != 0] # len(Z)/2]
[l.set_visible(False) for (i,l) in enumerate(ax.get_yticklabels()) if i % len(Z) != 0] # len(Z)/2]
xticks = ax.xaxis.get_major_ticks()
yticks = ax.yaxis.get_major_ticks()
# Now remove ticks except for first momentum label
# for m in range(pmin,pmax+1):
#     for z in range(zmin,zmax+1):
#         xticks[zmin*m+z].set_visible(False)

for p in range(pmin,pmax+1):
    for z in range(zmin,zmax):
        xticks[z+(p-1)*zmax].set_visible(False)
        yticks[z+(p-1)*zmax].set_visible(False)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
ax.xaxis.set_tick_params(pad=14,bottom=False)
ax.yaxis.set_tick_params(pad=10)
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_yticklabels(), ha="center")

# # apply offset transform to all x ticklabels.
# for label in ax.xaxis.get_majorticklabels():
#     label.set_transform(label.get_transform() + 0.5)



fig.savefig("h5Covariance.pdf",dpi=500,transparent=truthTransparent,\
            bbox_inches='tight',pad_inches=0)
plt.show()
