#!/usr/bin/python3

#####/dist/anaconda/bin/python

#####
# PLOT JACK DATA STORED IN AN H5 FILE
#####

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
import pylab # to save figures to file
import sys,optparse
from collections import OrderedDict
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
# from pitd_util import *
from common_fig import *

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-H", "--h5File", type="string", default="x:x",
                  help='H5 file(s) to plot <x>:<x> (default = "x:x")')
parser.add_option("--sysH5", type="string", default="",
                  help='Use sysH5 to set systematic error estimate (i.e. from diff. fit windows of matrix elements (default = "")')
parser.add_option("-J", action="store_true", default=False, dest="gaussianJK_PITD",
                  help='Include pITD computed by JK & co. using Gaussian smearing')
parser.add_option("-d", "--dtypeName", type="string", default="",
                  help='Datatype name(s) to access <d>:<d> (default = "d:d")')
parser.add_option("--amp", type="string", default="",
                  help='Amplitude name (default = "")')
parser.add_option("-s", "--singleFig", type="int", default=0,
                  help='If more than one h5 file, default to multiple panels or plot within one (default = 0)')
parser.add_option("-z", "--zRange", type="string", default="",
                  help='Min/Max zsep in h5 <zmin>.<zmax> (default = '')')
parser.add_option("-p", "--momRange", type="string", default="",
                  help='Min/Max Momenta in h5 <pmin>.<pmax> (default = '')')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-a", "--axesOverride", type="int", default=0,
                  help='Override vertical axes range in evo/match plots (default = 0)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')

# Parse the input arguments
(options, args) = parser.parse_args()



insertions = { 3: { 'Redstar': 'b_b1xDA__J1_T1pP', 'CK': None },
               8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK': 'insertion_gt' },
               11: { 'Redstar': 'a_a1xDA__J1_T1pM', 'CK': None} }

labels = { 'evoKernel': { 'Re': r'$\mathfrak{Re} B\otimes\mathfrak{%s}\left(\nu,z^2\right)$'%options.amp,
                          'Im': r'$\mathfrak{Im} B\otimes\mathfrak{%s}\left(\nu,z^2\right)$'%options.amp },
           'matchingKernel' : { 'Re': r'$\mathfrak{Re} L\otimes\mathfrak{%s}\left(\nu,z^2\right)$'%options.amp,
                                'Im': r'$\mathfrak{Im} L\otimes\mathfrak{%s}\left(\nu,z^2\right)$'%options.amp },
           'pitd': {'Re': r'$\mathfrak{Re}\ \mathfrak{%s}\left(\nu,z^2\right)$'%options.amp,
                    'Im': r'$\mathfrak{Im}\ \mathfrak{%s}\left(\nu,z^2\right)$'%options.amp},
           'itd': { 'Re': r'$\mathfrak{Re}\ \mathcal{Q}\left(\nu,\mu^2\right)$',
                    'Im': r'$\mathfrak{Im}\ \mathcal{Q}\left(\nu,\mu^2\right)$'} }


# Prefactor of convolution
# prefactor = (0.303*4/3)/(2*np.pi)
prefactor = 1.0

# Color according the z value
mainColors=['gray']
for dc in [ cm.turbo(d) for d in np.linspace(0,1,9) ]:
    mainColors.append(dc)


cfgs=options.cfgs
zmin, zmax=tuple(int(z) for z in options.zRange.split('.'))
pmin, pmax=tuple(int(p) for p in options.momRange.split('.'))



# Instantiate some figures
fig_real, axesR = plt.subplots(len(options.h5File.split(':')),1,sharex=True,figsize=(11,8)) # ,sharey=True)
fig_imag, axesI = plt.subplots(len(options.h5File.split(':')),1,sharex=True,figsize=(11,8)) # ,sharey=True)

#################################################################################
# If we are comparing to JK's pITD, make another set of figures
#################################################################################
if options.gaussianJK_PITD:
    fig_r_zoom = plt.figure(figsize=(11,8));
    axes_r_zoom = fig_r_zoom.gca()
    fig_i_zoom = plt.figure(figsize=(11,8));
    axes_i_zoom = fig_i_zoom.gca()
    axes_r_zoom.set_xlim([0,2.1])
    axes_i_zoom.set_xlim([0,1.78])
    axes_r_zoom.set_ylim([0.85,1.06])
    axes_i_zoom.set_ylim([0,0.3])
    axes_r_zoom.set_xlabel(r'$\nu$')
    axes_r_zoom.set_ylabel(r'$\mathfrak{Re}\ \mathfrak{M}\left(\nu,z^2\right)$')
    axes_i_zoom.set_xlabel(r'$\nu$')
    axes_i_zoom.set_ylabel(r'$\mathfrak{Im}\ \mathfrak{M}\left(\nu,z^2\right)$')
    


# Reset the figures if more than one h5 is passed and a single figure is desired
if len(options.h5File.split(':'))>1 and bool(options.singleFig):
    fig_real, axesR = plt.subplots(1,1,sharex=True,figsize=(11,8)) #(9.33,8))
    fig_imag, axesI = plt.subplots(1,1,sharex=True,figsize=(11,8))
    
fig_real.subplots_adjust(hspace=0.095)
fig_imag.subplots_adjust(hspace=0.095)
# fig_real.set_xlabel(r'$\nu$',fontsize=16)
# fig_imag.xlabel(r'$\nu$',fontsize=16)
if len(fig_real.get_axes()) > 1:
    axesR[-1].set_xlabel(r'$\nu$') #,fontsize=16)
    axesI[-1].set_xlabel(r'$\nu$') #,fontsize=16)
else:
    axesR.set_xlabel(r'$\nu$') #,fontsize=16)
    axesI.set_xlabel(r'$\nu$') #,fontsize=16)

    

# For potentially two datasets in same figure
symbs=['o','^']
shifts=[0.0,0.05]
####################################
# ACCESS FILE HANDLE(S)
####################################
for nH5, h5file in enumerate(options.h5File.split(':')):
    h5In = h5py.File(h5file,'r')
    dtypeName = options.dtypeName.split(':')[nH5]

    # # IF sysH5 is passed, use it to include a systematic error
    sysErrH5=None if options.sysH5=='' else h5py.File(options.sysH5,'r')



    for z in range(zmin,zmax+1):
        ztag="zsep00%d"%z
        for m in range(pmin,pmax+1):
            # ptag="pz%d"%m
            ptag="pf00%d_pi00%d"%(m,m)

            for comp in ["Re", "Im"]:

                ioffeTime = -1
                avgMat = 0.0; avgMatErr = 0.0
                avgMatSys = 0.0; avgMatErrSys=0.0
                for g in range(0,cfgs):
                    ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                          (insertions[options.insertion]['Redstar'],\
                                           ztag,ptag,comp,dtypeName)][g]

                    # Collect from sysH5 to set sys. error
                    try:
                        ioffeTime, matSys = sysErrH5['/%s/%s/%s/jack/%s/%s'%\
                                                     (insertions[options.insertion]['Redstar'],\
                                                      ztag,ptag,comp,dtypeName)][g]
                    except:
                        matSys = mat
                    #-------------------------------

                    avgMat += mat*prefactor
                    avgMatSys += matSys*prefactor
                    
                avgMat *= (1.0/cfgs)
                avgMatSys *= (1.0/cfgs)
                

                # Determine the error (and potentially include a systematic error estimate)
                for g in range(0,cfgs):
                    ioffeTime, mat = h5In['/%s/%s/%s/jack/%s/%s'%\
                                          (insertions[options.insertion]['Redstar'],\
                                           ztag,ptag,comp,dtypeName)][g]
                    avgMatErr += np.power( mat*prefactor - avgMat, 2)


                # Since only variance is plotted w/in this method, squared diff. btwn. avg matelems from h5In and sysErrh5 is added to avgMatErr
                avgMatErrSys += avgMatErr + np.power( avgMatSys - avgMat, 2)
                avgMatErrSys = np.sqrt( ((1.0*(cfgs-1))/cfgs)*avgMatErrSys )
                avgMatErr = np.sqrt( ((1.0*(cfgs-1))/cfgs)*avgMatErr )


                if comp == "Re":
                    if len(fig_real.get_axes()) > 1:
                        axesR[nH5].errorbar(ioffeTime, avgMat, yerr=avgMatErr, fmt='o',\
                                            color=mainColors[z],mec=mainColors[z],\
                                            mfc=mainColors[z],label=r'$z=%s$'%z)
                    else:
                        lab=r'$z=%s$'%z
                        # Include the systematic error bar, if sysErrh5 was passed
                        if sysErrH5 != None:
                            axesR.errorbar(ioffeTime+shifts[nH5], avgMat, yerr=avgMatErrSys,\
                                           fmt=symbs[nH5],color=mainColors[z],mec=mainColors[z],\
                                           mfc=mainColors[z],label=lab,alpha=0.6)

                        if nH5 == 1:
                            lab=None
                        for a in [axesR]: # , axes_r_zoom]:
                            a.errorbar(ioffeTime+shifts[nH5], avgMat, yerr=avgMatErr,\
                                       fmt=symbs[nH5],color=mainColors[z],mec=mainColors[z],\
                                       mfc=mainColors[z],label=lab)


                    
                            
                if comp == "Im":
                    if len(fig_real.get_axes()) > 1:
                        axesI[nH5].errorbar(ioffeTime, avgMat, yerr=avgMatErr, fmt='o',\
                                            color=mainColors[z],mec=mainColors[z],\
                                            mfc=mainColors[z],label=r'$z=%s$'%z)
                    else:
                        lab=r'$z=%s$'%z
                        # Include the systematic error bar, if sysErrh5 was passed
                        if sysErrH5 != None:
                            axesI.errorbar(ioffeTime+shifts[nH5], avgMat, yerr=avgMatErrSys,\
                                           fmt=symbs[nH5],color=mainColors[z],mec=mainColors[z],\
                                           mfc=mainColors[z],label=lab,alpha=0.6)
                            
                        if nH5 == 1:
                            lab=None
                        for a in [axesI]: # , axes_i_zoom]:
                            a.errorbar(ioffeTime+shifts[nH5], avgMat, yerr=avgMatErr,\
                                       fmt=symbs[nH5],color=mainColors[z],mec=mainColors[z],\
                                       mfc=mainColors[z],label=lab)


                        


for n, d in enumerate(options.dtypeName.split(':')):
    if len(fig_real.get_axes()) > 1:
        axesR[n].set_ylabel(labels[d]['Re'])
        axesI[n].set_ylabel(labels[d]['Im'])

        axesR[n].axhline(y=0.0,ls=':',color='gray')
        axesI[n].axhline(y=0.0,ls=':',color='gray')

        yrangeR = axesR[n].get_ylim()[1] - axesR[n].get_ylim()[0]
        yrangeI = axesI[n].get_ylim()[1] - axesI[n].get_ylim()[0]
    
        axesR[n].set_ylim([axesR[n].get_ylim()[0]-yrangeR*0.05, axesR[n].get_ylim()[1]+yrangeR*0.05])
        axesI[n].set_ylim([axesI[n].get_ylim()[0]-yrangeI*0.05, axesI[n].get_ylim()[1]+yrangeI*0.05])
        # Override default ylim settings
        if options.axesOverride == 1:
            # axesR[n].set_ylim([-30,30])
            # axesI[n].set_ylim([-30,30])

	    # axesR[n].set_ylim([0.97,1.05])
	    # axesI[n].set_ylim([-0.05,0.05])

            axesR[n].set_xlim([0,12.5])
            axesI[n].set_xlim([0,12.5])


    else:
        axesR.set_ylabel(labels[d]['Re'])
        axesI.set_ylabel(labels[d]['Im'])

        axesR.axhline(y=0.0,ls=':',color='gray')
        axesI.axhline(y=0.0,ls=':',color='gray')

        yrangeR = axesR.get_ylim()[1] - axesR.get_ylim()[0]
        yrangeI = axesI.get_ylim()[1] - axesI.get_ylim()[0]

        axesR.set_ylim([axesR.get_ylim()[0]-yrangeR*0.05, axesR.get_ylim()[1]+yrangeR*0.05])
        axesI.set_ylim([axesI.get_ylim()[0]-yrangeI*0.05, axesI.get_ylim()[1]+yrangeI*0.05])

        axesR.set_ylim([-0.4,1.05])
        axesI.set_ylim([-0.4,1.05])

	# axesR.set_ylim([0.97,1.0001])
        # axesI.set_ylim([-0.05,0.05])
        # dummy=10
        # axesR.set_xlim([-dummy,dummy])
        # axesI.set_xlim([-dummy,dummy])
        # axesR.set_ylim([-dummy,dummy])
        # axesI.set_ylim([-dummy,dummy])

        if options.axesOverride == 1:
            # # axesR.set_xlim([0,2.5])
            # axesR.set_ylim([0.81,1.02])
            # axesR.set_ylim([-0.5,1.1])


            ##### IF 'Y'
            axesR.set_xlim([-10,14])
            axesR.set_ylim([-0.25,1.05])
            axesI.set_xlim([-14,10])
            axesI.set_ylim([-0.8,0.8])
            
            # ##### IF 'R'
            # axesR.set_xlim([-10,14])
            # axesR.set_ylim([-0.15,0.3])
            # axesI.set_xlim([-14,10])
            # axesI.set_ylim([-0.25,0.25])
            
            # axesR.set_ylim([-0.4,1.05])
            # if int(options.momRange.split('.')[0]) < 0:
            #     axesR.set_xlim([-10,10]); axesI.set_xlim([-10,10])
            #     axesI.set_ylim([-0.8,0.8])
            # else:
            #     axesR.set_xlim([0,10]); axesI.set_xlim([-1,10])
            #     axesI.set_ylim([-0.05,0.8])

########################################
# IF JK H5 FILE IS PASSED, THEN PLOT IT
########################################
if options.gaussianJK_PITD:
    jkReal=np.loadtxt("H5s/JK-real_fine.dat")
    jkImag=np.loadtxt("H5s/JK-imag_fine.dat")
    
    for p in range(1,7):
        for z in range(1,9):

            ioffeTime = (2*np.pi/32)*z*p

            for a in [axesR, axes_r_zoom]:
                a.errorbar(ioffeTime+0.1,jkReal[(z-1)+8*(p-1),3],yerr=jkReal[(z-1)+8*(p-1),4],fmt='x',\
                               color=mainColors[z],mec=mainColors[z],mfc=mainColors[z])
            for a in [axesI, axes_i_zoom]:
                a.errorbar(ioffeTime+0.1,jkImag[(z-1)+8*(p-1),3],yerr=jkImag[(z-1)+8*(p-1),4],fmt='x',\
                           color=mainColors[z],mec=mainColors[z],mfc=mainColors[z])


    # Add dummy points to label the different datasets
    dumr = [axes_r_zoom.errorbar(-1,0.0,yerr=0.1,fmt='o',color='gray'),
            axes_r_zoom.errorbar(-2,0.0,yerr=0.1,fmt='x',color='gray')]
    dumi = [axes_i_zoom.errorbar(-1,0.0,yerr=0.1,fmt='o',color='gray'),
            axes_i_zoom.errorbar(-2,0.0,yerr=0.1,fmt='x',color='gray')]

    # Make a better font
    font = fm.FontProperties(family='serif',style='normal',size=15)
    axes_r_zoom.legend(dumr, ["This Work","JHEP 12 (2019) 081"],prop=font,\
                       framealpha=LegendFrameAlpha)
    axes_i_zoom.legend(dumi, ["This Work","JHEP 12 (2019) 081"],prop=font,\
                       framealpha=LegendFrameAlpha)



######################################################################
# IF SYSH5 IS PASSED
######################################################################


if len(fig_real.get_axes()) > 1:                
    handles, labels = axesR[-1].get_legend_handles_labels()
else:
    handles, labels = axesR.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend = fig_real.legend(by_label.values(), by_label.keys(),loc='upper right',\
                         framealpha=LegendFrameAlpha,fancybox=True,bbox_to_anchor=(0.9,0.89))
# this for upper left: bbox_to_anchor=(0.12,0.89)
# this for lower left: bbox_to_anchor=(0.12,0.1))
frame = legend.get_frame()
# frame.set_facecolor("#1b212c")
# frame.set_edgecolor("#1b212c")

if len(fig_imag.get_axes()) > 1:
    handles, labels = axesI[-1].get_legend_handles_labels()
else:
    handles, labels = axesI.get_legend_handles_labels() 
by_label = OrderedDict(zip(labels, handles))
legend = fig_imag.legend(by_label.values(), by_label.keys(),loc='upper left',\
                         framealpha=LegendFrameAlpha,fancybox=True, bbox_to_anchor=(0.12,0.89))
frame = legend.get_frame()





figOut=options.h5File.replace('.h5','')

fig_real.savefig(figOut+"_real%s.%s"%(suffix,form), dpi=400,pad_inches=0.0,\
                 transparent=truthTransparent,format=form)
fig_imag.savefig(figOut+"_imag%s.%s"%(suffix,form), dpi=400,pad_inches=0.0,\
                 transparent=truthTransparent,format=form)


for A in [axesR, axesI]:
    print(A.get_xlim()[0])
    print(A.get_xlim()[1])
    print(A.get_ylim()[0])
    print(A.get_ylim()[1])
    print("*****")


if options.gaussianJK_PITD:
    fig_r_zoom.savefig("viewh5-JKComp_real%s.%s"%(suffix,form), dpi=400,pad_inches=0.0,\
                       transparent=truthTransparent,format=form)
    fig_i_zoom.savefig("viewh5-JKComp_imag%s.%s"%(suffix,form), dpi=400,pad_inches=0.0,\
                       transparent=truthTransparent,format=form)


plt.show()
