#!/usr/bin/python3

import numpy as np
import sys,optparse
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm

# Parse command line options
usage = "usage: %prog [options] "
parser = optparse.OptionParser(usage) ;
parser.add_option("-l", "--lightBkgd", type="int", default=1,
                  help='Format figs for light (1) or dark (0) background (default = 1)')

(options, args) = parser.parse_args()


################
# INITIALIZE GLOBAL PROPERTIES OF FIGURES
################
# Finalize the figures
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams.update({'errorbar.capsize': 2})
plt.rc('xtick.major',size=10)
plt.rc('ytick.major',size=10)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
plt.rc('axes',labelsize=24)
suffix=''
form='pdf'
truthTransparent=False
FrameAlpha=0.8
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
    FrameAlpha=0.0
    # legendFaceColor="#1b212c"
    suffix='.dark'
    form='png'
    

hbarc=0.1973269804
a=0.094
L=32.0
# M=0.356 # pion GeV
# M=1.1230844097 # nucleon GeV
M=0.535 # nucleon lattice units


# Color pts based on t^2 value


def makePerp(q):
    tmp=[]
    for qq in Q:
        x=qq.split('.')[0]
        y=qq.split('.')[1]

        tmp.append((float(x)**2+float(y)**2)*((2*np.pi)/(a*L))*hbarc)

    return tmp
    
    
def dispersion(p):
    
    return 

Q=['0.0.1','0.0.-1','0.0.2','0.0.-2',\
   '0.1.0','0.1.1','0.1.-1','0.1.2','0.1.-2',\
   '1.0.0',\
   '1.1.0','1.1.1','1.1.-1','1.1.2','1.1.-2',\
   '2.0.1','2.0.-1','2.0.2','2.0.-2']

perp=makePerp(Q)

P1_z=[((2*np.pi)/(a*L))*n*hbarc for n in range(-7,7)]
P2_z=[((2*np.pi)/(a*L))*n*hbarc for n in range(-7,7)]

skewness_tmp=[]
T_tmp=[]


ptSize=20
fig=plt.figure(figsize=(10,8.4))
ax=fig.gca()
ax.scatter(0,0,s=ptSize)
for i in P1_z:
    for j in P2_z:

        skt=0.0

        # If skewness = Nan (i.e. pfz = -piz) then dump to zero
        if j == -i:
            skewness_tmp.append(0)
            skt=0
        else:
            skewness_tmp.append((1.0*(i-j))/(i+j))
            skt=(1.0*(i-j))/(i+j)


        for n,q in enumerate(Q):
            print(q)
            # If
            if float(q.split('.')[2])*((2*np.pi)/(a*L))*hbarc == j-i:
                # Now set T
                tvalue=2*(M**2)+2*i*j-((perp[n]**2)/2)\
                        -2*np.sqrt(M**2+i**2+((perp[n]**2)/4))\
                        *np.sqrt(M**2+j**2+((perp[n]**2)/4))
                T_tmp.append(tvalue)

                ax.scatter(skt,tvalue,marker='o',\
                           s=ptSize,alpha=0.75,\
                           color=cm.RdYlGn(int(tvalue*-72.85714285714286)))
                           # color=cm.nipy_spectral(int(tvalue*-72.85714285714286)))

            
# print skewness

skewness=np.unique(skewness_tmp)
T=np.unique(T_tmp)


ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$t=(p_i-p_f)^2\quad(\rm{GeV}^2)$')
# plt.ylim([-1.5,0.05])

fig.savefig("xi_vs_t_estimates%s.%s"%(suffix,form),dpi=600,\
            transparent=truthTransparent,\
            bbox_inches='tight',pad_inches=0.1,format=form)




# # New figures for 3d viewing
# fig3d=plt.figure(figsize=(10,8.4))
# ax3d=fig3d.add_subplot(projection='3d')
# ax3d.set_xlabel(r'$\nu_i$')
# ax3d.set_ylabel(r'$\nu_f$')
# ax3d.set_zlabel(r'$t$')


# zseps=np.linspace(0,8,9)
# pi=[ (2*np.pi/L)*n for n in range(0,7) ]
# pf=[ (2*np.pi/L)*n for n in range(0,7) ]

# nu_i=[]; nu_f=[]
# for p in pi:
#     for z in zseps:
#         nu_i.append(p*z)
# nu_f=nu_i

# # nu_i,nu_f = np.meshgrid(nu_i,nu_f)


# z=[np.cos(nu_i[n]*f) for n,f in enumerate(nu_f)]

# ax3d.plot_trisurf(nu_i,nu_f,z,color='red')



plt.show()


