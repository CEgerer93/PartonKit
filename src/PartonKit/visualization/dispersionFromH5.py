#!/usr/bin/python3
###############################################################
# READ IN DATA & FIT TO 2PT FUNCTIONS (H5) AND PLOT DISPERSION
###############################################################
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import sys,optparse
import pylab
import scipy.special as spec
from scipy import optimize as opt
sys.path.append('/home/colin/QCD/pseudoDists/analysis/py-scripts/pseudo')
import pitd_util
import fit_util
import corr_util
from common_fig import *
from common_util import LG


def evalDispersion(grnd,ap):
    return np.sqrt(grnd**2+ap**2)


usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);


parser.add_option("-f","--corrH5",type="str", default='',
                  help='H5 file containing 2pt correlator data (default = '')')
parser.add_option("-F","--corrFitH5",type="str", default='',
                  help='H5 file containing fits to 2pt correlator data (default = '')')
parser.add_option("-g", "--cfgs", type="int", default=0,
                  help='Ensem configs (default = 0)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-N", "--StoN", type="float", default=2.0,
                  help='Signal to noise cut-off (default = 2.0)')
parser.add_option("-S", "--showStoN", type="int", default=0,
                  help='Include subplot illustrating S/N ratios vs. T (default = 0)')
parser.add_option("-n", action="store_true", default=False, dest="normalizeByContDisp",
                  help='Normalize energies by continuum dispersion reln (default = False)')
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')

# Parse the input arguments
(options, args) = parser.parse_args()
cfgs=options.cfgs




fig=plt.figure(figsize=(8,6))
ax=fig.gca()
ax.set_xlabel(r'$ap$')
ax.set_ylabel(r'$aE_0$')
ax.set_xlim([-0.1,1.35])
# ax.set_xlim([-0.05,0.8])
# ax.set_ylim([0.5,1.35])
LGs={r'G_{1g}': {'mark': 's', 'cmap': cm.Greys_r, 'colors': None},\
     r'Dic_4': {'mark': '^', 'cmap': cm.Purples_r, 'colors': None},\
     r'Dic_2': {'mark': 'o', 'cmap': cm.Blues_r, 'colors': None},\
     r'Dic_3': {'mark': 'h', 'cmap': cm.Greens_r, 'colors': None},\
     r'C4_{nm0}': {'mark': 'x', 'cmap': cm.Oranges_r, 'colors': None},\
     r'C4_{nnm}': {'mark': '*', 'cmap': cm.Reds_r, 'colors': None}} # ,\
     # 'C2nmp': {'mark': '+', 'cmap': cm.viridis_r, 'colors': None}}

# Access the h5 files
h5Corr=h5py.File(options.corrH5,'r')
h5Fit=h5py.File(options.corrFitH5,'r')


tSeries={}
# Get all the tSeries of the data
for p,v in h5Corr.items():
    for k, op in v['t0_avg'].items():
        tmax = len(op['rows_11/real/data'][0])
        tSeries.update({p: "%s.1.%s"%(0,tmax-1)})

tfitSeries={}
# Get all the tfitSeries
for k,v in h5Fit['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real'].items():
    tmin, tmax = list(v.keys())[0].replace('tfit_','').split('-')
    tfitSeries.update({k: '%s.1.%s'%(tmin,tmax)})

# Capture ground state mass
grnd=0.0; grndErr=0.0


#################################
#      Run over the fits
#################################
energies={} # dictionary for energies
for n,p in enumerate(h5Fit['/exp{-E0*T}*{a+b*exp{-{E1-E0}*T}}/E0/mean/real'].keys()):

    pSqr=0.0
    for i in pitd_util.as_tuple(p):
        pSqr += i**2
    pSqr*=np.power(2*np.pi/options.Lx,2)
    
    # Remove 'p' prefix from momentum string so I don't have to change fit_util.py
    p=p.replace('p','')
    
    corrFit=fit_util.twoState(h5Fit,cfgs,p,p,'real',tfitSeries["p%s"%p],\
                              col='blue',z=None,gam=None)
    corrFit.parse()


    # If we've read ground-state, set grnd & grndErr
    if p == '000':
        grnd=corrFit.pAvg['E0']; grndErr=np.sqrt(corrFit.pCov['E0','E0'])

    # Store energies for later plotting
    energies.update({p: {'mom': np.sqrt(pSqr), 'avg': corrFit.pAvg['E0'], 'err': np.sqrt(corrFit.pCov['E0','E0']), 'LG': LG(p)}})

    # if p == '000':
    #     break

    


# Use ground-state mass to determine continuum dispersion relation
cont=np.linspace(0,1.5,400)
contE=[evalDispersion(grnd,i) for i in cont]
contEErr=[np.sqrt(np.power(((grnd**2)/(grnd**2+i**2)),2)*grndErr**2) for i in cont]

# If we are normalizing by dispersion relation, reset contE and contEErr
if options.normalizeByContDisp:
    contE=[evalDispersion(grnd,i)/evalDispersion(grnd,i) for i in cont]
    contEErr=[np.sqrt(np.power(((grnd**2)/(grnd**2+i**2)),2)*grndErr**2)/evalDispersion(grnd,i) for i in cont]


######################################################################
# Plot dispersion relation - optionally normalizing by central value #
######################################################################
ax.plot(cont,contE,color='red')
ax.fill_between(cont,[a+e for a,e in zip(contE,contEErr)],\
                [a-e for a,e in zip(contE,contEErr)],color='red',alpha=0.4,lw=0)
#------------------------------------------------------------------------------------------------------------------


######################################################################
# Plot all fitted energies - optionally normalizing by central value #
######################################################################
# Determine number of unique ap values
allAP = [ap['mom'] for ap in energies.values()]
# allAP = [(ap['LG'],ap['mom']) for ap in energies.values()]

print(allAP)

uniqAP = dict([(ap, allAP.count(ap)) for ap in np.unique(allAP)])

trackUniqAP = dict([(u, 0) for u in uniqAP])

print("Unique")
print(uniqAP)
print("Track")
print(trackUniqAP)



print(list(uniqAP.values()))
for n,group in enumerate(LGs.keys()):

    print(LGs[group]['cmap'])
    
    try:
        LGs[group]['colors'] = [ LGs[group]['cmap'](d) for d in np.linspace(0.2,1,list(uniqAP.values())[n])]
    except:
        print("Failed w/ %s"%group)
        print(n)
        print(list(uniqAP.values())[n])




for k,e in energies.items():

    offset = 0. if k == '000' else e['mom'] + 0.005*(trackUniqAP[e['mom']] - uniqAP[e['mom']]/2.0)
    
    thisE    = e['avg']/evalDispersion(grnd,e['mom']) if options.normalizeByContDisp else e['avg']
    thisEErr = e['err']/evalDispersion(grnd,e['mom']) if options.normalizeByContDisp else e['err']

    thisGrp = LG(k)

    # Convenience
    cnt=trackUniqAP[e['mom']]


    print(trackUniqAP)
    
    # print(LGs[thisGrp]['colors'])
    print("THIS CNT = %i"%cnt)
    print("THIS LG  = %s"%thisGrp)
    print("NUM COLORS FOR LG = %i"%(len(LGs[thisGrp]['colors'])))


    
    ax.errorbar(offset,thisE,yerr=thisEErr,fmt=LGs[thisGrp]['mark'],\
                mec='gray',mfc=None,\
                label=r'$\vec{p}=(%i,%i,%i)$'%(int(pitd_util.as_tuple(k)[0]),\
                                               int(pitd_util.as_tuple(k)[1]),\
                                               int(pitd_util.as_tuple(k)[2])))
    
    # try:
    #     ax.errorbar(offset,thisE,yerr=thisEErr,fmt=LGs[thisGrp]['mark'],\
    #                 mec=LGs[thisGrp]['colors'][cnt],mfc=None,\
    #                 label=r'$\vec{p}=(%i,%i,%i)$'%(int(pitd_util.as_tuple(k)[0]),\
    #                                                int(pitd_util.as_tuple(k)[1]),\
    #                                                int(pitd_util.as_tuple(k)[2])))
    # except:
    #     ax.errorbar(offset,thisE,yerr=thisEErr,fmt=LGs[thisGrp]['mark'],\
    #                 mec='gray',mfc=None,\
    #                 label=r'$\vec{p}=(%i,%i,%i)$'%(int(pitd_util.as_tuple(k)[0]),\
    #                                                int(pitd_util.as_tuple(k)[1]),\
    #                                                int(pitd_util.as_tuple(k)[2])))

    # Update how many times this ap has been seen
    trackUniqAP[e['mom']]+=1



##################################################
# ADD DUMMY POINTS TO INDICATE WHICH CUBIC GROUP #
##################################################
xmin, xmax = ax.get_xlim(); rangex=xmax-xmin
ymin, ymax = ax.get_ylim(); rangey=ymax-ymin

for n,k in enumerate(LGs.keys()):
    ax.scatter(xmin+0.545*rangex,ymin+0.915*rangey-n*0.05,marker=LGs[k]['mark'],c='gray',s=34)
    ax.text(xmin+0.47*rangex-n*0.0145,ymin+0.9*rangey-n*0.05,s=r'$%s\!:$'%k,fontsize=14,c='gray')

    
# ax.scatter(xmin+0.545*rangex,ymin+0.915*rangey,marker='s',c='gray',s=34)
# ax.text(xmin+0.47*rangex,ymin+0.9*rangey,s=r'$G_{1g}\!:$',fontsize=14,c='gray')

# ax.scatter(xmin+0.545*rangex,ymin+0.865*rangey,marker='^',c='gray',s=34)
# ax.text(xmin+0.4555*rangex,ymin+0.85*rangey,s=r'$Dic_4\!:$',fontsize=14,c='gray')

####################
# Manage the legend
####################
handles, labels = ax.get_legend_handles_labels()
customHandles=[]
customLabels=[]

for n,l in enumerate(labels):
    if l not in customLabels:
        handles[n].has_xerr=True; handles[n].has_yerr=False
        customLabels.append(l)
        customHandles.append(handles[n])

ax.legend(customHandles,customLabels,\
          loc='upper left', ncol=2,handletextpad=0.1,columnspacing=0.1,\
          framealpha=LegendFrameAlpha,fontsize=11)
    

suffix=''
form='pdf'
if options.lightBkgd == 0:
    suffix='.dark'
    form='png'

plt.savefig("dispersion%s.%s"%(suffix,form),\
            dpi=400,transparent=truthTransparent,bbox_inches='tight',format=form)



plt.show()
