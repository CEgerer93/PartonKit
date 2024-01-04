#!/usr/bin/python3
############
# Estimate mass renormalization of Wilson line by computing
#         ln[ M(z+a,a,Pz) / M(z,a,Pz) ]
############
import numpy as np
import optparse
from util.fit_util import SR
from util.pitd_util import toStr
import util.common_fig

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-f","--fitSR",type="str", default='',
                  help='H5 file containing summed ratio fit results (default = '')')
parser.add_option("--cfgs", type="int", default=0,
                  help='Ensemble configs (default = 0)')
parser.add_option("-t","--fitTSeries", type="str", default='x.x.x:x.x.x',
                  help='tmin.step.tmax of fit (default = x.x.x:x.x.x)')
parser.add_option("--pf", type="str", default='x.x.x',
                  help='Final Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("--pi", type="str", default='x.x.x',
                  help='Initial Momenta <px>.<py>.<pz> (default = x.x.x)')
parser.add_option("-c","--component", type="str", default='real',
                  help='Read component (default = real)')
parser.add_option("-s", action="store_true", default=False, dest="showFig",
                  help='Put figure to screen')
parser.add_option("--pdfH5", action="store_true", default=False, dest="pdfH5",
                  help='Read fit results from h5 file that does not include snk/src row indices - i.e. from a pITD analysis (default = false)')
parser.add_option("--fwd", action="store_true", default=False, dest="fwdPDF",
                  help='Determine effective mass renormalization from forward PDF (default = False)')


# Parse the input arguments
(options, args) = parser.parse_args()

pf=toStr(options.pf)
pi=toStr(options.pi)


tmin, tstep, tmax = options.fitTSeries.split('.')

##########
# Figure
##########
figX=plt.figure(figsize=(11,8)); axX=figX.gca()
figY=plt.figure(figsize=(11,8)); axY=figY.gca()
fig4=plt.figure(figsize=(11,8)); ax4=fig4.gca()
for a in [axX, axY, ax4]:
    a.set_xlabel(r'$z/a$')
    a.set_ylabel(r'$\delta m_{\rm eff}$')
    a.set_ylim([-0.5,1.5])
    a.set_xticks(np.linspace(0,8,9))

################################
# Add data from h5 file to plot
################################
Z=np.arange(1,9)

for ng,gamma,col,ax in [(8,'gamma_4','b',ax4)] if options.fwdPDF else [(1,'gamma_x','r',axX), (2,'gamma_y','k',axY), (8,'gamma_4','b',ax4)]:

    lineTypes=['-',':','--','-.']
    for rowf in [1,2]:
        for rowi in [1,2]:
            if options.fwdPDF and ( rowf != rowi ): continue
            
            fitZ0=fit_util.SR(options.fitSR,options.cfgs,"pf%s"%pf,"pi%s"%pi,rowf,rowi,\
                              "zsep000",ng,options.component,\
                              int(tmin),int(tmax),pdfH5=options.pdfH5)
            fitZ0.parse()
            
            effMat={ 0: fitZ0.pAvg['b']}
            
            
            for z in Z:
                if z == 0: continue
                
                zsep="00%i"%z
                print("zsep%s"%zsep)
                # Construct a summed ratio fit instance
                fitSR=fit_util.SR(options.fitSR,options.cfgs,"pf%s"%pf,"pi%s"%pi,rowf,rowi,\
                                  "zsep%s"%zsep,ng,options.component,\
                                  int(tmin),int(tmax),pdfH5=options.pdfH5)
                fitSR.parse()
                
                effMat.update({z: fitSR.pAvg['b']})
                
                
            thisLineStyle=lineTypes.pop()
            ax.plot(Z,[-np.log(effMat[z]/effMat[z-1]) for z in Z],\
                    col,label=r'$\%s\ \mu_f,\mu_i=%i,%i$'%(gamma,rowf,rowi),\
                    ls=thisLineStyle)


############################################
# Add text indicating momenta/displacements
############################################
for a in [axX,axY,ax4]:
    yrange=a.get_ylim()[1]-a.get_ylim()[0]
    a.set_title(r'$\vec{p}_f=\left(%s,%s,%s\right)\quad\leftarrow\quad\vec{p}_i=\left(%s,%s,%s\right)$'\
                %(tuple(p for p in (options.pf+'.'+options.pi).split('.'))),fontsize=18)
    a.legend()

figX.savefig("eff-mass-renorm_gamma-x_pf%s_pi%s.%s.pdf"%(options.pf,options.pi,options.component))
figY.savefig("eff-mass-renorm_gamma-y_pf%s_pi%s.%s.pdf"%(options.pf,options.pi,options.component))
fig4.savefig("eff-mass-renorm_gamma-4_pf%s_pi%s.%s.pdf"%(options.pf,options.pi,options.component))
plt.show()
