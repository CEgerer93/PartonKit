#!/usr/bin/python3

'''
Take the difference in jackknife bins of two hdf5 files of identical directory trees
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
import sys,optparse
from collections import OrderedDict
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
sys.path.append('/home/colin/QCD/pseudoDists')
from common_fig import *
from common_util import *


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-H", "--firstH5", type="string", default="",
                  help='First H5 file to consider (default = "")')
parser.add_option("-I", "--secondH5", type="string", default="",
                  help='Second H5 file to consider (default = "")')
parser.add_option("-d", "--dtypeName", type="string", default="",
                  help='Datatype name to access (default = "x")')
parser.add_option("-c", "--comp", type="string", default="",
                  help='Which component <Re -or- Im> to view difference of (default='')')


# Parse the input arguments
(options, args) = parser.parse_args()


# Init a figure/axis
fig=plt.figure(figsize=(11,8))
ax=fig.gca()
ax.set_ylabel(r'${\rm %s } \mathfrak{M}_{\rm SVD}\left(\nu,z^2\right)-{\rm %s } \mathfrak{M}_{\rm old}\left(\nu,z^2\right)$'%(options.comp,options.comp))
ax.set_xlabel(r'$\nu$')


# Set a color scheme to color by z/a
mainColors=['gray']
for dc in [ cm.turbo(d) for d in np.linspace(0,1,9) ]:
    mainColors.append(dc)




h1=h5py.File(options.firstH5)
h2=h5py.File(options.secondH5)


for ins in h1.keys():
    h5Dirac = h1.get(ins)
    
    for zsep in h5Dirac.keys():
        # Convenience - zsep as an int
        z=as_tuple(zsep)[-1]
        h5ZSep = h5Dirac.get(zsep)
        for mom in h5ZSep.keys():
            h5Mom = h5ZSep.get(mom)



            h1_jack=h5Mom.get('jack/%s/pitd'%options.comp)
            h2_jack=h2.get('/%s/%s/%s/jack/%s/pitd'%(ins,zsep,mom,options.comp))


            ioffe=h1_jack[0,0]
            diff_jack=h1_jack[:,1]-h2_jack[:,1]


            cfgs=len(diff_jack)
            diff_avg=sum(diff_jack)/(1.0*cfgs)
            diff_err=0.0


            for v in diff_jack:
                diff_err+=np.power(v-diff_avg,2)
            diff_err=np.sqrt( ((1.0*(cfgs-1))/cfgs)*diff_err )

            
            print("%.4f +/- %.9f"%(diff_avg,diff_err))


            lab=r'$z=%s$'%z
            ax.errorbar(ioffe,diff_avg,yerr=diff_err,fmt='o',
                        color=mainColors[z],mec=mainColors[z],\
                        mfc=mainColors[z],label=lab)
            

# Grab unique labels
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(),loc='upper right',\
                   framealpha=LegendFrameAlpha,fancybox=True)


fig.savefig('diff_%s.pdf'%options.comp)
plt.show()
