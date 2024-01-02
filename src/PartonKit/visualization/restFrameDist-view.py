#!/usr/bin/python3
#######################################################
# READ IN PSEUDO-ITD DATA AND SEE IF DGLAP IS PRESENT #
#######################################################
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys,optparse
import pylab
import scipy.special as spec
from scipy import optimize as opt


import pitd_util



usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-f","--fitSR",type="str", default='',
                  help='H5 file containing summed ratio fits (default = '')')
parser.add_option("--cfgs", "--cfgs", type="int", default=0,
                  help='Ensem configs (default = 0)')
parser.add_option("--pf", type="str", default='x.x.x',
                  help='Final Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--pi", type="str", default='x.x.x',
                  help='Initial Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-T","--tFitRange",type="str", default='x.x',
                  help='T fit range <tmin>.<tmax> (default = "x.x")')
parser.add_option("-c","--component", type="str", default='',
                  help='Read component (default = Re)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')

# Parse the input arguments
(options, args) = parser.parse_args()



##################
# Parse Args
##################
cfgs=options.cfgs
pf=pitd_util.toStr(options.pf)
pi=pitd_util.toStr(options.pi)

compStr=''
if options.component == 'real':
    compStr='Re'
elif options.component == 'imag':
    compStr='Im'



# Initialize figure properties
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['errorbar.capsize'] = 2.0
# plt.rcParams['ytick.direction'] = 'out'
plt.rc('xtick.major',size=10)
plt.rc('ytick.major',size=10)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
plt.rc('axes',labelsize=24)
truthTransparent=False
LegendFrameAlpha=1.0 #0.8
legendFaceColor="white"
suffix=''
form='pdf'
# Optionally swap default black labels for white
if options.lightBkgd == 0:
    truthTransparent=True
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    # plt.rcParams['legend.edgecolor'] = "#1b212c"
    plt.rc('axes',edgecolor='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('text',color='white')
    LegendFrameAlpha=0.0
    legendFaceColor="#1b212c"
    suffix='.dark'
    form='png'
    

# Init the figure
fig=plt.figure(figsize=(11,8))
ax=fig.gca()

##########
# Label axis - omit component prefix if no component is specified - i.e. both will be shown
##########
ax.set_xlabel(r'$z_3/a$')
if options.component == '':
    ax.set_ylabel(r'$\mathcal{M}_{%s}\left(0,z^2\right)$'%\
                  pitd_util.insertions[options.insertion]['Dirac'])
else:
    ax.set_ylabel(r'$\mathfrak{%s}\ \mathcal{M}\left(0,z^2\right)$'%compStr)


# Modify axes, etc for case where pgitds are under consideration
if options.pf != options.pi:
    ax.set_xlabel(r'$\nu\equiv\left(\nu_f+\nu_i\right)/2$')
    ax.set_ylabel(r'$\mathfrak{%s}\mathcal{M}\left(\nu,\xi,t;z^2\right)$'%compStr)
    ax.text(0.5,0.75,r'$\xi=%.4f$'%pitd_util.skewness(options.pf,options.pi))
    ax.text(0.4,0.75,r'$t=?$')



    
symbs={'real': 'o', 'imag': '^'}

############
# Read the h5 file
############
h=h5py.File(options.fitSR,'r')

# List of components to plot
if options.component=='':
    comps=[('real',symbs['real']), ('imag',symbs['imag'])]
else:
    comps=[(options.component,symbs[options.component])]


# Plot according to comps list
for c in comps:
    disp=h['/a+bT/b/bins/%s/pf%s_pi%s'%(c[0],pf,pi)]

    for d in disp.keys():
        pDist=disp['%s/gamma-%d'%(d,options.insertion)].\
            get('tfit_%s-%s'%(options.tFitRange.split('.')[0],options.tFitRange.split('.')[1]))

        zvec='0.0.%s'%d[-1]

        x,y,z=tuple(int(i) for i in zvec.split('.'))
        
        ioffeIni=pitd_util.ioffeTime(options.pi,zvec,options.Lx)
        ioffeFin=pitd_util.ioffeTime(options.pf,zvec,options.Lx)
        ioffe=(ioffeIni+ioffeFin)/2.0



        # Compute average and error from jackknife bins
        mean=0.0; err=0.0
        for n, d in enumerate(pDist):
            mean+=(d/(1.0*len(pDist)))

        # jk_pDist=pitd_util.makeJks(pDist)
        # jkAvg_pDist=pitd_util.makeAvgJks(jk_pDist)
        
        ###### ?????
        jk_pDist=pitd_util.matelem(pDist,False)
        
        # for n, j in enumerate(jkAvg_pDist):
        for n, j in enumerate(jk_pDist.data):
            err+=np.power(j-mean,2)
        err=np.sqrt(((cfgs-1)/(1.0*cfgs))*err)
    
        # plt.errorbar(ioffe,pDist[0],yerr=pDist[1],fmt='o')
        
        ###############
        # PLOT WRT TO Z
        ###############
        plt.errorbar(z,mean,yerr=err,fmt=c[1],color=pitd_util.mainColors[z],\
                     mec=pitd_util.mainColors[z],\
                     mfc=pitd_util.mainColors[z],label=r'$z=%s$'%z)
        
        ###############
        # PLOT WRT TO IOFFE-TIME
        ###############
        # plt.errorbar(ioffe,mean,yerr=err,fmt='o',color=pitd_util.mainColors[z],\
            #              mec=pitd_util.mainColors[z],\
            #              mfc=pitd_util.mainColors[z],label=r'$z=%s$'%z)
  



plt.show()
