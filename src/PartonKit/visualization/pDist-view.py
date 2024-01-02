#!/usr/bin/python3
#######################################################
# READ IN PSEUDO-ITD DATA AND SEE IF DGLAP IS PRESENT #
#######################################################
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import sys,optparse
import pylab
import scipy.special as spec
from scipy import optimize as opt
from collections import OrderedDict

import pitd_util
sys.path.append('/home/colin/QCD/pseudoDists')
import fit_util

from common_fig import *


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-f","--amps",type="str", default='',
                  help='H5 file containing amplitudes (default = '')')
parser.add_option("--fit2pt",type="str", default='',
                  help='H5 file containing 2pt fits (default = '')')
parser.add_option("--cfgs", "--cfgs", type="int", default=0,
                  help='Ensem configs (default = 0)')
parser.add_option("--pf", type="str", default='x.x.x',
                  help='Final Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--pi", type="str", default='x.x.x',
                  help='Initial Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--momShell", type="str", default='0.0.0',
                  help='Equal ini/fin momenta w/in shell |px|.|py|.|pz| - pf/pi options then ignored (default = 0.0.0)')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-T","--tFitRanges",type="str", default='x.x',
                  help='T fit range <tmin>.<tmax>, colon delimited (default = "x.x")')
parser.add_option("-c","--component", type="str", default='Re',
                  help='Read component (default = Re)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-a", "--aLatt", type="float", default=0.0,
                  help='Lattice spacing (default = 0.0)')
parser.add_option("--wrtZ", action="store_true", default=False, dest="wrtZ",
                  help='Plot pDist w.r.t. zsep instead of Ioffe-time')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')
parser.add_option("-s", action="store_true", default=False, dest="showPlots",
                  help='Display plots to screen (default = False)')
parser.add_option("-d", action="store_false", default=True, dest="useDispRelnForMomTrans",
                  help='Do not use dispersion relation in computing momentum transfer (default = True)')

# Parse the input arguments
(options, args) = parser.parse_args()



##################
# Parse Args
##################
cfgs=options.cfgs
pf=pitd_util.toStr(options.pf)
pi=pitd_util.toStr(options.pi)

########
# To make momenta w/in shell
########
def shell():
    moms=''
    for px in range(-int(options.momShell.split('.')[0]),int(options.momShell.split('.')[0])+1):
        for py in range(-int(options.momShell.split('.')[1]),int(options.momShell.split('.')[1])+1):
            for pz in range(-int(options.momShell.split('.')[2]),int(options.momShell.split('.')[2])+1):
                moms+='%i.%i.%i:'%(px,py,pz)
    return moms[:-1]
    
                

compStr=''
if options.component == 'real':
    compStr='Re'
elif options.component == 'imag':
    compStr='Im'

# Diff symbols for diff tfit ranges
symbs=['o','^','s']
# For horizontal shifting
STEP=0.0
        

##########################################
# CREATE FIGURES & SET INITIAL AXES LABELS
##########################################
ampDict={}
# Off-forward amplitude h5 has M,L
if options.pf != options.pi:
    for amp in ['A%i'%i for i in range(1,11)]:
        ampDict.update({amp: { 'fig': None, 'ax': None}})
# Helicity amplitude h5 has M,N,R,Y
else:
    ampDict.update({'R': { 'fig': None, 'ax': None}})
    ampDict.update({'Y': { 'fig': None, 'ax': None}})

for k,v in ampDict.items():
    v['fig']=plt.figure(figsize=(11,8))
    v['ax']=v['fig'].gca()
    if options.wrtZ:
        v['ax'].set_xlabel(r'$z/a$')
        v['ax'].set_xticks(np.linspace(-8,8,9))
    else:
        v['ax'].set_xlabel(r'$\nu$')
    if pf != pi:
        v['ax'].set_ylabel(r'$\mathfrak{%s}\ \mathcal{%s}_{%s}\left(\nu,\xi,t;z^2\right)$'\
                           %(compStr,k[0],k[1:]))
    else:
        v['ax'].set_ylabel(r'$\mathfrak{%s}\ \mathcal{%s}\left(\nu,z^2\right)$'%(compStr,k))
        if options.wrtZ:
            v['ax'].set_ylabel(r'$\mathfrak{%s}\ \mathcal{%s}\left(0,z^2\right)$'%(compStr,k))
#--------------------------------------------------------------------------

print(ampDict)
####################################
# OPEN 2PT FIT H5 TO ACCESS Ef,Ei,m
####################################
h2pt=h5py.File(options.fit2pt,'r')

# Make the keys for each momenta
tSeries={}; twoPtFits={}
if pf != 'xxx' and pi != 'xxx' and options.momShell == '0.0.0':
    tSeries={'000': None, pf: None, pi: None}
else:
    for p in shell().split(':'):
        tSeries.update({'%s%s%s'%(p.split('.')[0],p.split('.')[1],p.split('.')[2]): None})

# For each momenta in tSeries, read set min/max tsep and read the 2-state fit
for k in tSeries.keys():
    tmin2pt, tmax2pt = list(h2pt['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real/p%s'%k].\
                            keys())[0].replace('tfit_','').split('-')
    tSeries[k]='%s.1.%s'%(tmin2pt,tmax2pt)
    twoPtFits.update({k : fit_util.twoState(h2pt,cfgs,k,k,'real',tfitSeries=tSeries[k],col='k')})

# Parse the 2-state fits for each key
for k,v in twoPtFits.items():
    v.parse()
    print(k)



# Set average Ef,Ei,m for momentum transfer evaluation  - RELEVANT ONLY FOR PGITD PLOTS
Ef=0.0; Ei=0.0; m=0.0
if options.momShell == '0.0.0':
    Ef=twoPtFits[pf].pAvg['E0'] 
    Ei=twoPtFits[pi].pAvg['E0']
    m=twoPtFits['000'].pAvg['E0']
h2pt.close()
#--------------------------------------------------------------------------------------


##################################################
# OPEN 3PT H5 FILE W/ AMPLITUDES ONCE AND FOR ALL
##################################################
h=h5py.File(options.amps,'r')
nAmps=len(h.keys()) # Number of distinct amplitudes contained w/in h5 file
    


##################################################################
# RUN THROUGH ALL DISPLACEMENTS FOR EACH COMBINATION OF MOMENTA
##################################################################
momenta=[(options.pf,options.pi)]
if options.momShell != '0.0.0': momenta=[(p,p) for p in shell().split(':')]


for PF,PI in momenta:
    for A in h.keys():
        print(A)

        path='/%s/bins/%s/pf%s_pi%s'%\
                   (A,options.component,pitd_util.toStr(PF),pitd_util.toStr(PI))
        try:
            disp=h[path]
        except:
            raise Exception("In h5 = %s, cannot find path %s"%(h,path))
            
        for d in disp.keys():
            for N,T in enumerate(options.tFitRanges.split(':')):

                pDist=disp['%s/gamma-%d'%(d,options.insertion)].\
                    get('tfit_%s-%s'%(T.split('.')[0],T.split('.')[1]))

                print(pDist)
                
                # # RESCALE BY DIVIDING MATELEM BY PREFACTOR OF pDIST
                # pDistRescale=np.divide(pDist,prefactor)
                # print(pDistRescale)
                
                
                zvec='0.0.%s'%d.replace('zsep00','')
                x,y,z=tuple(int(i) for i in zvec.split('.'))
                
                ioffeIni=pitd_util.ioffeTime(PI,zvec,options.Lx)
                ioffeFin=pitd_util.ioffeTime(PF,zvec,options.Lx)
                ioffe=(ioffeIni+ioffeFin)/2.0
                
                
                
                jk_pDist=pitd_util.matelem(pDist,False)
                # Compute average and error from jackknife bins
                mean=0.0; err=0.0
                for n,dat in enumerate(jk_pDist.data):
                    mean+=(dat/(1.0*len(jk_pDist.data)))
                    
                # for n, j in enumerate(jkAvg_pDist):
                for n, j in enumerate(jk_pDist.data):
                    err+=np.power(j-mean,2)
                err=np.sqrt(((cfgs-1)/(1.0*cfgs))*err)
                
                
                ##############################
                # PLOT WRT TO Z or IOFFE-TIME
                ##############################
                STEP=(abs(pitd_util.ioffeTime(PI,'0.0.1',options.Lx))+
                      abs(pitd_util.ioffeTime(PF,'0.0.1',options.Lx)))/2
                wrt=ioffe

                # STEP=0.0
                
                if options.wrtZ:
                    wrt=z
                    STEP=1
                    
                wrt += (N*0.1*STEP) # horizontally shift multiple datasets by 10% of stepsize
            
                ampDict[A]['ax'].errorbar(wrt,mean,yerr=err,fmt=symbs[N],\
                                          color=pitd_util.mainColors[z],\
                                          mec=pitd_util.mainColors[z],\
                                          mfc=pitd_util.mainColors[z],label=r'$z=%s$'%z)


##############################################################################
# INCLUDE TEXT FOR MOMENTUM/DISP & MODIFY AXES + INCLUDE XI/T TEXT FOR PGITDS
##############################################################################
for K,V in ampDict.items():
    ax=V['ax']
    yrange=ax.get_ylim()[1]-ax.get_ylim()[0]
    xtent=ax.get_xlim()[1]-ax.get_xlim()[0]

    # Extend xrange so legend will fit nicely
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]+3*STEP])

    fooBuff=0
    # fooBuff=-0.6 # for vertical shifting of text, until some automatic implementation
    if pf != pi: # formerly 0.56*xtent below
        ax.text(ax.get_xlim()[0]+0.7*xtent,(0.92+fooBuff)*yrange+ax.get_ylim()[0],\
                r'$\vec{p}_f=\left(%s,%s,%s\right)$'%\
                tuple(p for p in options.pf.split('.')),fontsize=18)
        ax.text(ax.get_xlim()[0]+0.7*xtent,(0.87+fooBuff)*yrange+ax.get_ylim()[0],\
                r'$\vec{p}_i=\left(%s,%s,%s\right)$'%\
                tuple(p for p in options.pi.split('.')),fontsize=18)
        ax.text(ax.get_xlim()[0]+0.7*xtent,(0.8+fooBuff)*yrange+ax.get_ylim()[0],\
                r'$\xi=%.4f$'%pitd_util.skewness(options.pf,options.pi),fontsize=18)
        ax.text(ax.get_xlim()[0]+0.7*xtent,(0.75+fooBuff)*yrange+ax.get_ylim()[0],\
                r'$t=%.4f\thinspace{\rm GeV}^2$'%\
                pitd_util.momTrans(Ef,options.pf,Ei,options.pi,m,\
                                   options.Lx,options.aLatt,options.useDispRelnForMomTrans),\
                fontsize=18)

    else:
        # plt.text(3.5,0.4*yrange+ax.get_ylim()[0],r'$\vec{p}=\left(%s,%s,%s\right)$'%\
        #          tuple(p for p in options.pi.split('.')),fontsize=18)
        continue

        
#------------------------------------------------------------------------------------




# Fetch all the unique labels on each fig, and make the legends
for fig, ax in [tuple([V['fig'],V['ax']]) for V in ampDict.values()]:
    # Get the handles/labels
    handles, labels = ax.get_legend_handles_labels()
    Ncol=1

    # Catch indices of first unique displacement
    catch=[labels.index("%s"%z) for z in np.unique(labels)]

    # labelsNew=[]; handlesNew=[]
    labelsNew=[labels[n] for n in catch]
    handlesNew=[handles[n] for n in catch]


    hlDict={}
    for n in range(0,len(labelsNew)):
        hlDict.update({labelsNew[n]: handlesNew[n]})

    # Sort the legend keys on displacement
    labelsNewSort=sorted(labelsNew,key=lambda s: int(s.replace('$z=','').replace('$','')))

    
    handlesNewSort=[]
    for k in labelsNewSort:
        handlesNewSort.append(hlDict[k])


    # Turn errorbars in legend to be horizontal
    for h in handlesNew:
        h.has_xerr=True; h.has_yerr=False

    by_label=OrderedDict(zip(labelsNewSort,handlesNewSort))
    ax.legend(by_label.values(),by_label.keys(),framealpha=LegendFrameAlpha,ncol=Ncol,loc=1)
        

suffix=''
wrt=''
form='pdf'
dpi=200
if options.lightBkgd == 0:
    suffix='.dark'
    form='png'
if options.wrtZ:
    wrt='.wrtZ'


for K in ampDict.keys():
    ampDict[K]['fig'].savefig("pDist_%s_g%i_pf%s_pi%s_%s%s%s.%s"%\
                              (K,options.insertion,pf,pi,compStr,wrt,suffix,form),\
                              dpi=dpi,transparent=truthTransparent,bbox_inches='tight',format=form)

if options.showPlots:
    plt.show()
