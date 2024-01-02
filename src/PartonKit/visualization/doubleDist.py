#!/usr/bin/python3.8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import pylab
import sys,optparse
from collections import OrderedDict

import pheno
import gpd_utils as GPD
from common_fig import *





path=mpath.Path([[0,1],[1,0],[0,-1],[-1,0],[0,1]])
patch=mpatches.PathPatch(path,facecolor='none')

print(path)
print(patch)


fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.set_zlim3d([0,4])
ax.set_xlabel(r'$\beta$',labelpad=10)
ax.set_ylabel(r'$\alpha$',labelpad=10)

a=np.linspace(-1,1,100)
b=a

DD=np.zeros((len(a),len(a)))
for nb,bi in enumerate(b):
    for na,ai in enumerate(a):
        dd=GPD.doubleDistribution(bi,ai,5)

        if np.abs(bi) <= 1 and np.abs(ai) <= 1-np.abs(bi):
            DD[nb,na]=dd.profile()
            if DD[nb,na] > ax.get_zlim()[1]:
                DD[nb,na] = ax.get_zlim()[1]
        else:
            DD[nb,na]=-1

b,a = np.meshgrid(b,a)



masking=np.abs(b)+np.abs(a) > 1

DDprime=np.ma.masked_where(masking,DD)


surface=ax.plot_surface(b,a,DDprime, rstride=1, cstride=1, linewidth=0,\
                        alpha=0.95, antialiased=False, cmap=cm.get_cmap('rainbow'),\
                        clip_on=True,clip_path=patch)

# surface.set_clip_on(True)
# surface.set_clim(vmin=0,vmax=4)


plt.show()
