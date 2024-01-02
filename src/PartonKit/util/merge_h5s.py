#!/usr/bin/python3

#####/dist/anaconda/bin/python

#####
# PLOT JACK DATA STORED IN AN H5 FILE
#####

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab # to save figures to file
import sys,optparse
from collections import OrderedDict
from pitd_util import mainColors

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-H", "--h5File", type="string", default="",
                  help='H5 file (default = "")')
parser.add_option("-G", "--h5File2", type="string", default="",
                  help='H5 file2 (default = "")')
parser.add_option("-d", "--dtypeName", type="string", default="",
                  help='Datatype name(s) to access <d>:<d> (default = "d:d")')
parser.add_option("-z", "--zRange", type="string", default="",
                  help='Min/Max zsep in h5 <zmin>.<zmax> (default = '')')
parser.add_option("-Z", "--finalZRange", type="string", default="",
                  help='Final Min/Max zsep in h5 <zmin>.<zmax> (default = '')')
parser.add_option("-p", "--momRange", type="string", default="",
                  help='Min/Max Momenta in h5 <pmin>.<pmax> (default = '')')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
# parser.add_option("-a", "--axesOverride", type="int", default=0,
#                   help='Override vertical axes range in evo/match plots (default = 0)')
# parser.add_option("-l", "--lightBkgd", type="int", default=1,
#                   help='Format figs for light (1) or dark (0) background (default = 1)')

insertions = { 8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK' : 'insertion_gt' } }
labels = { 'evoKernel': { 'Re': r'$\mathfrak{Re} B\otimes\mathfrak{M}\left(\nu,z^2\right)$',
                          'Im': r'$\mathfrak{Im} B\otimes\mathfrak{M}\left(\nu,z^2\right)$' },
           'matchingKernel' : { 'Re': r'$\mathfrak{Re} L\otimes\mathfrak{M}\left(\nu,z^2\right)$',
                                'Im': r'$\mathfrak{Im} L\otimes\mathfrak{M}\left(\nu,z^2\right)$' },
           'pitd': {'Re': r'$\mathfrak{Re} \mathfrak{M}\left(\nu,z^2\right)$',
                    'Im': r'$\mathfrak{Im} \mathfrak{M}\left(\nu,z^2\right)$'},
           'itd': { 'Re': r'$\mathfrak{Re} \mathcal{Q}\left(\nu,\mu^2\right)$',
                    'Im': r'$\mathfrak{Im} \mathcal{Q}\left(\nu,\mu^2\right)$'} }


# Prefactor of convolution
# prefactor = (0.303*4/3)/(2*np.pi)
prefactor = 1.0


# Parse the input arguments
(options, args) = parser.parse_args()
cfgs=options.cfgs
# zmin=int(options.zRange.split('.')[0])
# zmax=int(options.zRange.split('.')[1])
pmin=int(options.momRange.split('.')[0])
pmax=int(options.momRange.split('.')[1])




merge={}


####################################
# ACCESS FILE HANDLE(S)
####################################
for nH5, h5file in enumerate(options.h5File.split(':')):
    h5In = h5py.File(h5file,'r')
    dtypeName = options.dtypeName.split(':')[0]



    thisZSep=options.zRange.split(':')[nH5]
    print(thisZSep)
    zmin=int(thisZSep.split('.')[0])
    zmax=int(thisZSep.split('.')[1])


    # print(zmin)
    # print(zmax)

    

    for z in range(zmin,zmax+1):
        ztag="zsep%d"%z

        merge.update({ztag: {}})
        
        for m in range(pmin,pmax+1):
            ptag="pz%d"%m

            merge[ztag].update({ptag: {}})
            
            for comp in ["Re", "Im"]:

                merge[ztag][ptag].update({comp: {'ioffe': -1, 'data': None}})
                
                ioffeTime = -1
                mat=[]
                
                for g in range(0,cfgs):
                    merge[ztag][ptag][comp]['ioffe'], m = h5In['/%s/%s/%s/jack/%s/%s'%(insertions[options.insertion]['Redstar'],ztag,ptag,comp,dtypeName)][g]

                    mat.append(m)
                    


                merge[ztag][ptag][comp]['data'] = mat


# print(merge)



#################################
# MAKE THE NEW H5 FILE
#################################
h5out = h5py.File('the-merged.h5','w')
dtypeName = options.dtypeName.split(':')[0]

h5Dirac = h5out.get(insertions[options.insertion]['Redstar'])
if h5Dirac == None:
    h5Dirac = h5out.create_group(insertions[options.insertion]['Redstar'])


# for z in range(1,17):
# for n,z in enumerate(options.zRange.split(':')):
for z in range(int(options.zRange.split(':')[0].split('.')[0]),
               int(options.zRange.split(':')[-1].split('.')[-1])+1):
# for z in range(options.finalZRange.split(':')[0],options.finalZRange.split(':')[1]+1):
    ztag="zsep%d"%z #.split('.')[0])
    h5sep = h5Dirac.get(ztag)
    if h5sep == None:
        h5sep = h5Dirac.create_group(ztag)
        

    for m in range(pmin,pmax+1):
        ptag="pz%d"%m
        h5mom = h5sep.get(ptag)
        if h5mom == None:
            h5mom = h5sep.create_group(ptag+"/jack")


        for comp in ["Re", "Im"]:
            h5comp = h5mom.get(comp)
            if h5comp == None:
                h5comp = h5mom.create_group(comp)



            h5outDat = h5comp.create_dataset(dtypeName,(cfgs,2), 'd')

            for g in range(0,cfgs):
                h5outDat[g,0] = merge[ztag][ptag][comp]['ioffe']
                h5outDat[g,1] = merge[ztag][ptag][comp]['data'][g]


h5out.close()

