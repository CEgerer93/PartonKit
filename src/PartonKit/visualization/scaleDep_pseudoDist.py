#!/usr/bin/python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab # to save figures to file
import sys,optparse

sys.path.append("/home/colin/QCD/pseudoDists")
import h5_utils
import common_fig

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-H", "--h5File", type="string", default="",
                  help='H5 file(s) to plot <x>:<x> (default = "x:x")')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-d", "--dtype", type="string", default="",
                  help='Datatype name(s) to access <d>:<d> (default = "d:d")')
parser.add_option("--log", action="store_true", default=False, dest="logScale",
                  help='Plot z-dependence on log scale')
parser.add_option("--asFuncOfZ2", action="store_true", default=False, dest="asFuncOfZ2",
                  help='Plot rpITD as function of z^2')

(options, args) = parser.parse_args()


h=h5py.File(options.h5File,'r')


##############
# Define color disp mapping
##############
mainColors=['gray']
for dc in [ cm.viridis_r(d) for d in np.linspace(0,1,9) ]:
    mainColors.append(dc)


##############
# Sorting key
##############
def sortKey(elem):
    return int(elem[4:])


###
# Find all the unique ioffeTimes in h5
###
uniqZ=[z for z in h['/a_a1xDA__J1_T1pM'].keys()]
uniqP=[p for p in h['/a_a1xDA__J1_T1pM/zsep000'].keys()]




ioffe={}

comp='Im'
for z in uniqZ:
    for p in uniqP:

        itTmp=h['/a_a1xDA__J1_T1pM/%s/%s/jack/%s'%(z,p,comp)].get('pitd')[0][0]

        # If this Ioffe time exists already, append the {z,p} combo
        if itTmp in ioffe:
            ioffe[itTmp]['z'].append(z)
            ioffe[itTmp]['p'].append(p)
        else:
            ioffe.update({itTmp: {'z': [z], 'p': [p]}})




    
for it in ioffe.keys():
    print("%.4f   %i"%(it,len(ioffe[it]['z'])))

    

    if len(ioffe[it]['z']) <= 2 or it == 0:
        continue
    else:
        fig=plt.figure(figsize=(12,10)); ax=fig.gca()
        ax.set_xlabel(r'$z$') if not options.asFuncOfZ2 else ax.set_xlabel(r'$z^2$')
        ax.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{Y}\left(\nu,z^2\right)$')
        # ax.semilogx()
        if options.logScale:
            ax.set_xscale('log')

        
        # Sort on zsep, so legend shows increasing z
        ioffe[it]['z'].sort(key=sortKey)
        for n,v in enumerate(ioffe[it]['z']):

            print(v)
            print(ioffe[it]['p'][n])
            zint=int(v[4:])
            
            dum, avg, err = h5_utils.reader(h,options.cfgs,v,ioffe[it]['p'][n],\
                                              comp,options.dtype,curr='a_a1xDA__J1_T1pM')

            if options.asFuncOfZ2:
                ax.errorbar(np.power(zint,2),avg,yerr=err,fmt='o',\
                            capsize=3,color=mainColors[zint],
                            label=r'$z/a=%i$'%zint)
            else:
                ax.errorbar(zint,avg,yerr=err,fmt='o',\
                            capsize=3,color=mainColors[zint],
                            label=r'$z/a=%i$'%zint)


        # Get max range in x & y
        xmin=ax.get_xlim()[0]; xmax=ax.get_xlim()[1]
        xrang=np.abs(xmax-xmin)

        ax.set_xlim([0,16])
        ax.set_ylim([-0.25,1.02])
        ymin=ax.get_ylim()[0]; ymax=ax.get_ylim()[1]
        yrang=np.abs(ymax-ymin)

        print("(x,y) = (%.4f, %.4f)"%(xmin+0.2*xrang,ymin+0.2*yrang))

        ax.text(xmin+0.2*xrang,ymin+0.2*yrang,r'$\nu=%.4f$'%it,fontsize=18)
        ax.legend(loc='best',framealpha=common_fig.LegendFrameAlpha,fancybox=True)
        fig.savefig("scaleDep_pITD.it_%.4f.pdf"%it)

plt.show()
