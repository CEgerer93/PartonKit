#!/dist/anaconda/bin/python

# from /u/home/cegerer/src/sharedfuncs import *
from matplotlib import cm
from functools import reduce
from sharedfuncs import *
import h5py
import re
import sys
sys.path.append('/home/colin/QCD/pseudoDists')
import fit_util

# Color according the z value
mainColors=['gray']
# for own in ['blue','red','green','purple','orange','magenta',(0.1,1,0.1),'black','gray',\
#             'teal','indigo','brown','salmon','dodgerblue','darkgreen','crimson','cyan']:
#     mainColors.append(own)
for dc in [ cm.turbo(d) for d in np.linspace(0,1,17) ]:
    mainColors.append(dc)
# for dc in [ cm.brg(d) for d in np.linspace(0,1,17) ]:
#     mainColors.append(dc)



#####################
# Redstar insertions
#####################
insertions = { 3: { 'Redstar': 'b_b1xDA__J1_T1pP', 'CK': None, 'Dirac': '\gamma_1\gamma_2'},
               8: { 'Redstar': 'b_b0xDA__J0_A1pP', 'CK': 'insertion_gt', 'Dirac': '\gamma_4' },
               11: { 'Redstar': 'a_a1xDA__J1_T1pM', 'CK': None, 'Dirac': '\gamma_z\gamma_5'},
               -1: { 'Redstar': 'b_b0xDA__J0_A1pP'} }
#------------------------------------------------------------------------------------------------


##############
# CONSTANTS
##############
HBARC=0.1973269804
#-------------------------


# Make a tuple
def as_tuple(s):
    ints=re.compile('-?\d') # Match group of digits that could possibly be preceeded by a minus sign
    return tuple(int(i) for i in ints.findall(s))
        
        
# Average a list
def avg(d):
    return reduce(lambda a, b: a + b, d) / len(d)

# Global function to determine ioffe-time
def ioffeTime(p,z,L):
    it=0.0

    P=p; Z=z
    # Check is p/z passed as tuples
    if type(p) is not tuple:
        P=tuple(int(i) for i in p.split('.'))
    if type(z) is not tuple:
        Z=tuple(int(i) for i in z.split('.'))

    for n,i in enumerate(P):
        it+=float(i)*float(Z[n])
    return ((2*np.pi)/L)*it


# Skewness - assuming only z-momenta != 0
def skewness(pf,pi):
    return (1.0*(int(pi.split('.')[2])-int(pf.split('.')[2])))/\
        (int(pi.split('.')[2])+int(pf.split('.')[2]))

# Momentum transfer
def momTrans(Ef,pf,Ei,pi,m,L,a,computeFromDispReln=False):
    dot=0.0
    for n,v in enumerate(pf.split('.')):
        dot+=int(v)*int(pi.split('.')[n])
    dot*=np.power((2*np.pi/L),2)

    if not computeFromDispReln:
        return 2*(m**2-Ef*Ei+dot)*(np.power(HBARC/a,2))
    else:
        dotPf=0.0; dotPi=0.0
        for n in range(0,3):
            dotPf+=int(pf.split('.')[n])*int(pf.split('.')[n])
            dotPi+=int(pi.split('.')[n])*int(pi.split('.')[n])
        dotPf*=np.power((2*np.pi/L),2)
        dotPi*=np.power((2*np.pi/L),2)
        Ef=np.sqrt(m**2+dotPf)
        Ei=np.sqrt(m**2+dotPi)
        return 2*(m**2-Ef*Ei+dot)*(np.power(HBARC/a,2))


# Global function to make zsep string
def makeAdatZSep(z):
    s=''
    for i in range(0,z):
        s+='3'
    return s

# Global function to cast momentum to string
def mom2Str(x,y,z):
    return str(x)+str(y)+str(z)

def toStr(vec,sp='.'):
    vstr=''
    for i in vec.split(sp):
        vstr+=str(i)
    return vstr
    

# Global function to make mom_type
def mom_type(m):
    tmp=[]
    [tmp.append(i) for i in m]
    for n, m in enumerate(tmp):
        if int(m) < 0:
            m = str(int(m)*-1)

    # Now sort and return
    tmp.sort(reverse=True)
    return "%s%s%s"%(tmp[0],tmp[1],tmp[2])            


###############################################################################################
# Parse fit results of an npt corr, with file name generated using adat/lib/hadron headers
###############################################################################################
class nptOp:
    def __init__(self,t,iso,row,mom,op,disp=''):
        self.t = t
        self.iso = iso
        self.row = 'r'+str(row)
        self.mom = str(mom)
        self.op = op+'__'+mom_type(self.mom)
        self.disp = disp
        self.name = ''

    def setName(self):
        if self.name != '': # reset the name if something non-trivial is read
            self.name = ''
        for s in [self.t, self.iso, self.row, self.mom, self.op, self.disp]:
            self.name += '%s%s'%(s,',')
        self.name = self.name.rstrip(",")
#-----------------------------------------------------------------------------------------------


class matelem:
    def __init__(self,data,forceNullM):
        self.data = data  # Passed data
        self.params = {}  # Dict of all fitted parameters
        self.n = None     # Number of unique fitted parameters
        self.cfgs = 0
        self.jkparams = {}
        self.avgjkparams = {}
        self.forceNullM = forceNullM # zero matrix element if comp = 2 and pz = 0

    def parseParams(self):
        uniqParams=[]
        for n, d in enumerate(self.data):
            if d[0] not in uniqParams:
                self.params.update({ d[0]: [] })
                self.jkparams.update({ d[0]: None })
                self.avgjkparams.update({ d[0]: None })

        # Now set number of unique fitted parameters & # cfgs fitted
        self.n = len(self.params)
        self.cfgs = len(self.data)/self.n
        
    def getParams(self):
        for n, d in enumerate(self.data):
            # Fill from fit results, zeroing if comp = 2 and pz = 0
            if self.forceNullM:
                self.params[d[0]].append(0.0)
            else:
                self.params[d[0]].append(d[1])

    # Jackknife the fitted parameters
    def jk(self):
        for k, v in self.params.items():
            self.jkparams[k] = makeJks(self.params[k])

    # Form average of fitted parameters per jackknife sample
    def avgjk(self):
        for k, v in self.params.items():
            self.avgjkparams[k] = makeAvgJks(self.jkparams[k])



class correlator:
    def __init__(self,o1,o2,o3,mType,defSearchDir):
        self.snk=o1
        self.ins=o2
        self.src=o3
        self.mType=mType # res matrix element type - so far, summedRatio
        self.dsd=defSearchDir # default search directory
        if type(self.snk) is str:
            print(type(self.snk))
            self.name=self.snk.name+"."+self.ins.name+"."+self.src.name

        self.amplitude = { 1: None, 2: None }

    def resRead(self):
        for a, m in self.amplitude.items():
            # Force matelem to zero if we are reading imaginary data at pz=0 (i.e. comp = 2, pz = 0)
            zeroOut=False
            # if a == 2 and self.ins.disp == '':
            #     zeroOut=True

            compTag=""
            if a == 1:
                compTag="RE"
            if a == 2:
                compTag="IM"
            
            # Extract matrix element from fit results - potentially zeroing out
            # self.amplitude[a] = matelem(np.genfromtxt("%s%s.%s.comp_%d.RES.dat"%\
            self.amplitude[a] = matelem(np.genfromtxt("%s%s.%s.%s_fit.dat"%\
                                                      (self.dsd,self.name,self.mType,compTag),dtype=None),\
                                        forceNullM=zeroOut)        



class ratio:
    def __init__(self,N,D):
        self.nr=N.amplitude[1].avgjkparams['B']
        self.ni=N.amplitude[2].avgjkparams['B']
        self.dr=D.amplitude[1].avgjkparams['B']
        self.di=D.amplitude[2].avgjkparams['B']
        self.avgjkratio=np.zeros(len(self.nr))
        self.R = { 1: {'ensem': [], 'jks': [], 'avgjks': []},
                   2: {'ensem': [], 'jks': [], 'avgjks': []} }
        # self.R = { 1: {'jks': [], 'avg': 0.0, 'err': 0.0 },
        #            2: {'jks': [], 'avg': 0.0, 'err': 0.0 } }


    # def fillEnsem(self):
    #     for g in range(0,len(self.nr)):
    #         self.R[1]['ensem'].append( self.nr[g]

    # Form the ratio per jackknife ensemble average
    def fillJkAvgs(self):
        for g in range(0,len(self.nr)):
            self.R[1]['jks'].append((self.nr[g]*self.dr[g]+self.ni[g]*self.di[g])/(self.dr[g]**2+self.di[g]**2))
            self.R[2]['jks'].append((self.dr[g]*self.ni[g]-self.nr[g]*self.di[g])/(self.dr[g]**2+self.di[g]**2))

    def avg(self):
        for j in range(0,len(self.R[1]['jks'])):
            self.R[1]['avg'] += self.R[1]['jks'][j]
            self.R[2]['avg'] += self.R[2]['jks'][j]
        self.R[1]['avg']*=(1.0/len(self.R[1]['jks']))
        self.R[2]['avg']*=(1.0/len(self.R[2]['jks']))



class pitdOld:
    def __init__(self,B,Bnd,N,Nnd,B_nIT=None,Bnd_nIT=None,N_nIT=None,Nnd_nIT=None,matStr='A'):
        self.B_r=generic1D(B.amplitude[1].params[matStr])
        self.B_i=generic1D(B.amplitude[2].params[matStr])
        self.N_r=generic1D(N.amplitude[1].params[matStr])
        self.N_i=generic1D(N.amplitude[2].params[matStr])
        self.Bnd_r=generic1D(Bnd.amplitude[1].params[matStr])
        self.Bnd_i=generic1D(Bnd.amplitude[2].params[matStr])
        self.Nnd_r=generic1D(Nnd.amplitude[1].params[matStr])
        self.Nnd_i=generic1D(Nnd.amplitude[2].params[matStr])

        self.B_r_nIT=None; self.B_i_nIT=None; self.Bnd_r_nIT=None; self.Bnd_i_nIT=None
        self.N_r_nIT=None; self.N_i_nIT=None; self.Nnd_r_nIT=None; self.Nnd_i_nIT=None


        # Assuming here if one of B_nIT,Bnd_nIT,N_nIT,Nnd_nIT is not NoneType,
        # then make more generic1D instances
        if B_nIT is not None:
            self.B_r_nIT=generic1D(B_nIT.amplitude[1].params[matStr])
            self.B_i_nIT=generic1D(B_nIT.amplitude[2].params[matStr])
            self.Bnd_r_nIT=generic1D(Bnd_nIT.amplitude[1].params[matStr])
            self.Bnd_i_nIT=generic1D(Bnd_nIT.amplitude[2].params[matStr])
            self.N_r_nIT=generic1D(N_nIT.amplitude[1].params[matStr])
            self.N_i_nIT=generic1D(N_nIT.amplitude[2].params[matStr])
            self.Nnd_r_nIT=generic1D(Nnd_nIT.amplitude[1].params[matStr])
            self.Nnd_i_nIT=generic1D(Nnd_nIT.amplitude[2].params[matStr])

    
            print("B4 B_r = %.12f"%np.mean(self.B_r.dat))
            print("B4 B_r_nIT = %.12f"%np.mean(self.B_r_nIT.dat))
            for g in range(0,self.B_r.dim):
                self.B_r.dat[g] += self.B_r_nIT.dat[g]; self.B_r.dat[g] /= 2.0;
                self.B_i.dat[g] -= self.B_i_nIT.dat[g]; self.B_i.dat[g] /= 2.0;
                self.N_r.dat[g] += self.N_r_nIT.dat[g]; self.N_r.dat[g] /= 2.0;
                self.N_i.dat[g] -= self.N_i_nIT.dat[g]; self.N_i.dat[g] /= 2.0;
                self.Bnd_r.dat[g] += self.Bnd_r_nIT.dat[g]; self.Bnd_r.dat[g] /= 2.0;
                self.Bnd_i.dat[g] -= self.Bnd_i_nIT.dat[g]; self.Bnd_i.dat[g] /= 2.0;
                self.Nnd_r.dat[g] += self.Nnd_r_nIT.dat[g]; self.Nnd_r.dat[g] /= 2.0;
                self.Nnd_i.dat[g] -= self.Nnd_i_nIT.dat[g]; self.Nnd_i.dat[g] /= 2.0;

            print("AFTER = %.12f"%np.mean(self.B_r.dat))


        self.R_r=generic1D(list(np.zeros(self.B_r.dim)))
        # self.R_r.dim=self.B_r.dim
        self.R_i=generic1D(list(np.zeros(self.B_r.dim)))
        # self.R_i.dim=self.B_r.dim

        self.rDist = { 1: {'jks': [], 'avg': 0.0, 'err': 0.0 },
                       2: {'jks': [], 'avg': 0.0, 'err': 0.0 } }


    # def fillJkAvgs(self):
    #     for g in range(0,len(self.B_r)):
    #         self.pITD[1]['jks'].append((self.B_r[g]*self.Nnd_r[g])/(self.Bnd_r[g]*self.N_r[g]))
    #         self.pITD[2]['jks'].append((self.B_i[g]*self.Nnd_r[g])/(self.Bnd_r[g]*self.N_r[g]))
    # def avg(self):
    #     for c, v in self.pITD.items():
    #         for j in range(0,len(self.B_r)):
    #             v['avg'] += v['jks'][j]
    #         v['avg'] *= (1.0/len(self.B_r))
    # def err(self):
    #     for c, v in self.pITD.items():
    #         dum=0.0
    #         dumc=len(self.B_r)
    #         for j in range(0,dumc):
    #             dum += np.power( v['jks'][j] - v['avg'], 2)
    #         v['err'] = np.sqrt( ((dumc-1)/(1.0*dumc))*dum)

            
    def avg(self):
        for c in [self.B_r, self.B_i, self.N_r, self.N_i,\
                  self.Bnd_r, self.Bnd_i, self.Nnd_r, self.Nnd_i]:
            c.average()

    # ######################
    # # DO NOT REJACKKNIFE!
    # ######################
    # def jk(self):
    #     for c in [self.B_r, self.B_i, self.N_r, self.N_i,\
    #               self.Bnd_r, self.Bnd_i, self.Nnd_r, self.Nnd_i]:
    #         c.makeJks()
    # def jkAvg(self):
    #     for c in [self.B_r, self.B_i, self.N_r, self.N_i,\
    #               self.Bnd_r, self.Bnd_i, self.Nnd_r, self.Nnd_i]:
    #         c.makeAvgJks()
    # def biasCorrect(self):
    #     self.R_r.average(); self.R_i.average()
    #     self.R_r.removeBias(); self.R_i.removeBias()
    # #---------------------------------------------------------------

    def makeReducedDist(self):
        for j in range(0,self.B_r.dim):
            # n.b. Do not rejackknife - but if you're not conviced swap dat for jkAvg in this method
            self.R_r.dat[j]=(self.B_r.dat[j]*self.Nnd_r.dat[j]*self.N_r.dat[j]\
                             +self.B_i.dat[j]*self.Nnd_r.dat[j]*self.N_i.dat[j])\
                             /(self.N_r.dat[j]*self.Bnd_r.dat[j]*self.N_r.dat[j]\
                               +self.N_i.dat[j]*self.Bnd_r.dat[j]*\
                               self.N_i.dat[j])
            
            self.R_i.dat[j]=(self.B_i.dat[j]*self.Nnd_r.dat[j]*self.N_r.dat[j]\
                             -self.B_r.dat[j]*self.Nnd_r.dat[j]*self.N_i.dat[j])\
                             /(self.N_r.dat[j]*self.Bnd_r.dat[j]*self.N_r.dat[j]\
                               +self.N_i.dat[j]*self.Bnd_r.dat[j]*\
                               self.N_i.dat[j])

            # Keep track of individual jackknife bins as well
            self.rDist[1]['jks'].append(self.R_r.dat[j])
            self.rDist[2]['jks'].append(self.R_i.dat[j])

    def getReducedDist(self):
        # n.b. Do not rejackknife - if not convinced, uncomment two pairs of lines in this method
        # self.R_r.makeJks(); self.R_i.makeJks()
        # self.R_r.makeAvgJks(); self.R_i.makeAvgJks()

        # The average
        for j in range(0,self.B_r.dim):
            self.rDist[1]['avg'] += self.R_r.dat[j]
            self.rDist[2]['avg'] += self.R_i.dat[j]
        self.rDist[1]['avg'] *= (1.0/self.B_r.dim)
        self.rDist[2]['avg'] *= (1.0/self.B_r.dim)
        
        # The error
        for j in range(0,self.B_r.dim):
            # self.rDist[1]['err'] += np.power( self.R_r.jkAvg[j] - self.rDist[1]['avg'], 2)
            # self.rDist[2]['err'] += np.power( self.R_i.jkAvg[j] - self.rDist[2]['avg'], 2)
            self.rDist[1]['err'] += np.power( self.R_r.dat[j] - self.rDist[1]['avg'], 2)
            self.rDist[2]['err'] += np.power( self.R_i.dat[j] - self.rDist[2]['avg'], 2)
        self.rDist[1]['err']=np.sqrt( ((self.R_r.dim-1)/(1.0*self.R_r.dim))*self.rDist[1]['err'])
        self.rDist[2]['err']=np.sqrt( ((self.R_r.dim-1)/(1.0*self.R_r.dim))*self.rDist[2]['err'])
#--------------------------------------------------------------------------------------------       


###########################################
###########################################
########## NEWEST PITD STUFF ##############
###########################################
###########################################
class pitdKey:
    def __init__(self,pf,pi,z):
        self.pf=pf
        self.pi=pi
        self.z=z

    def key(self):
        return "%s_%s,%s"%(self.pf,self.pi,self.z)

    def pfpi(self):
        return "%s_%s"%(self.pf,self.pi)



class common_pseudoDist:
    def extract(self,extAmp=None,extMom=None):
        ampsToExtract=self.amplitudes if extAmp is None else {extAmp: self.amplitudes[extAmp]}

        for AMP,V in ampsToExtract.items():
        # for AMP,V in self.amplitudes.items():

            for p,pval in self.h['/%s/bins/real'%AMP].items():
                # Next iteration is extMom is set and p != extMom
                if extMom is not None and p != extMom: continue

                for z in pval.keys():
                    print(p)
                    key=pitdKey(p.split('_')[0],p.split('_')[1],z)
                    print(key.key())
                    V.update({key.key(): {'real': {}, 'imag': {}}})

                    for comp,data in V[key.key()].items():
                        data.update({'avg': 0.0, 'err': 0.0, 'jks': None})

                        # Access data from h5
                        data['jks']=list(self.h['/%s/bins/%s/%s_%s/%s/gamma-%i/tfit_%i-%i'%\
                                                (AMP,comp,p.split('_')[0],p.split('_')[1],\
                                                 z,self.gam,self.tmin,self.tmax)])

                        
                        # # For helicity PDF (need to clean this)
                        # if AMP == 'R':
                        #     dataY=list(self.h['/Y/bins/%s/%s_%s/%s/gamma-%i/tfit_%i-%i'%\
                        #                       (comp,p.split('_')[0],p.split('_')[1],\
                        #                        z,self.gam,self.tmin,self.tmax)])
                        
                        # Zero out if comp=imag and pf=pi=0 or z are zero
                        if comp == 'imag' and ( ( key.pf == "pf000" and key.pi == "pi000" ) or key.z == "zsep000" ):
                            print("Canceling for component %s with (pf,pi,z) = (%s,%s,%s)"%(comp,key.pf,key.pi,key.z))
                            data['jks']=[0.0 for n in range(0,self.cfgs)]
                            # if AMP == 'R':
                            #     dataY=[0.0 for n in range(0,self.cfgs)]
                        

                        # if AMP != 'R':
                        #     data['avg']=sum(data['jks'][:])/len(data['jks'])

                        data['avg']=sum(data['jks'][:])/len(data['jks'])
                        

                        # if AMP == 'R':
                        #     for j in range(0,len(data['jks'])):
                        #         data['avg']+=(data['jks'][j]-dataY[j])/len(data['jks'])
                        #         # print("DATA MINUS = %.5f"%(data['jks'][j]-dataY[j]))
                        
                        for j in range(0,len(data['jks'])):
                            data['err']+=np.power(data['jks'][j]-data['avg'],2)
                        data['err']=np.sqrt( ((len(data['jks'])-1)/(1.0*len(data['jks'])))*data['err'] )
                    
    def applyParity(self,L=32):
        '''
        Apply parity in z to a pseudo-distribution
        Resets data...
        '''
        for AMP,V in self.amplitudes.items():
            for p,pval in self.h['/%s/bins/real'%AMP].items():

                ioffeMap={}
                for z in pval.keys():

                    pf,pi = p.split('_')
                    key=pitdKey(pf,pi,z)
                    ioffeFin = ioffeTime(as_tuple(pf),as_tuple(z),L)
                    ioffeIni = ioffeTime(as_tuple(pi),as_tuple(z),L)
                    ioffeAvg = (ioffeFin+ioffeIni)/2.0

                    ioffeMap.update({ioffeAvg: key.key()})

                itPairs=[]
                for i in ioffeMap.keys():
                    if -1*i in ioffeMap:
                        itPairs.append([i,-i])

                # Now get all unique pairs
                itPair=[]
                for s in itPairs:
                    if s not in itPair:
                        itPair.append(s)


                for s in itPair:
                    print(s)
                    # print("%s   %s"%(ioffeMap[s[0]],ioffeMap[s[1]]))

                #sys.exit()
        
###############################################################################################
# Support for visualizing polynomial fit applied to reduced pseudo-ITD data
###############################################################################################
class pitdPoly:
    def __init__(self,a,b,c,nu):
        self.paramOrder = {0: 'a', 1: 'b', 2: 'c', 3: 'rChi2'}


    # Evaulate the polynomial fit
    def func(self,comp,nu):
        if comp == 0:
            return 1.0+self.avgParams[0]['a']*self.nu**2+self.avgParams[0]['b']*self.nu**4\
                +self.avgParams[0]['c']*self.nu**6
        if comp == 1:
            return self.avgParams[1]['a']*self.nu+self.avgParams[1]['b']*self.nu**3\
                +self.avgParams[1]['c']*self.nu**5


    # Evaluate the derivative of polynomial fit
    def dfunc(self,comp,nu):
        partials={} # local dict for ordering partials
        for k in self.avgParams[0].keys():
            partials.update({k: 0.0})

        if comp == 0:
            partials['a'] = self.nu**2
            partials['b'] = self.nu**4
            partials['c'] = self.nu**6
            partials['nu'] = 2*self.avgParams[0]['a']*self.nu+4*self.avgParams[0]['b']*self.nu**3\
                             +6*self.avgParams[0]['c']*self.nu**5
        if comp == 1:
            partials['a'] = self.nu
            partials['b'] = self.nu**3
            partials['c'] = self.nu**5
            partials['nu'] = self.avgParams[1]['a']+3*self.avgParams[1]['b']*self.nu**2\
                             +5*self.avgParams[1]['c']*self.nu**4

        error=0.0
        for i in self.avgParams[comp]:
            for j in self.avgParams[comp]:
                error+=partials[i]*self.cov[comp][(i,j)]*partials[j]
        return np.sqrt(error)

            
    # Plot the fit - stopping at ioffeCut
    def plotFit(self,axR,axI,ioffeCut):
        axR.plot(self.nu,self.func(0,self.nu),color=mainColors[self.zsep])
        axR.fill_between(self.nu,self.func(0,self.nu)+self.dfunc(0,self.nu),\
                         self.func(0,self.nu)-self.dfunc(0,self.nu),color=mainColors[self.zsep],\
                         alpha=0.25,where=self.nu[:]<ioffeCut)
        axI.plot(self.nu,self.func(1,self.nu),color=mainColors[self.zsep])
        axI.fill_between(self.nu,self.func(1,self.nu)+self.dfunc(1,self.nu),\
                         self.func(1,self.nu)-self.dfunc(1,self.nu),color=mainColors[self.zsep],\
                         alpha=0.25,where=self.nu[:]<ioffeCut)
#----------------------------------------------------------------------------------------------


###############################################################################################
# Class to manage polynomial fit applied to reduced pseudo-ITD data - inherits pitdPoly methods
###############################################################################################
class polyFit(pitdPoly):
    def __init__(self,r,i,zlabel):
        self.realJks=r['jks']
        self.imagJks=i['jks']
        self.dumPoly = pitdPoly(0,0,0,0)
        self.cfgs = len(self.realJks)
        self.avgParams={0: {}, 1: {}}
        self.cov={0: {}, 1: {}}

        self.nu=np.linspace(0,20,800)
        self.zsep=zlabel
        

    def getAvgParams(self):
        for k,v in self.dumPoly.paramOrder.items():
            avgR=0.0
            avgI=0.0
            for g in range(0,self.cfgs):
                avgR += float(self.realJks[g][k])
                avgI += float(self.imagJks[g][k])
            avgR*=(1.0/(1.0*self.cfgs))
            avgI*=(1.0/(1.0*self.cfgs))

            # Add these average values
            self.avgParams[0].update({v: avgR})
            self.avgParams[1].update({v: avgI})

    def getParamCov(self):
        for lk,lv in self.dumPoly.paramOrder.items():
            for rk,rv in self.dumPoly.paramOrder.items():
                key=(lv,rv)
                dumR=0.0
                dumI=0.0
                for g in range(0,self.cfgs):
                    dumR += (float(self.realJks[g][lk])-self.avgParams[0].get(lv))*\
                            (float(self.realJks[g][rk])-self.avgParams[0].get(rv))
                    dumI += (float(self.imagJks[g][lk])-self.avgParams[1].get(lv))*\
                            (float(self.imagJks[g][rk])-self.avgParams[1].get(rv))

                dumR *= ((1.0*(self.cfgs-1))/self.cfgs)
                dumI *= ((1.0*(self.cfgs-1))/self.cfgs)
                
                self.cov[0].update({key: dumR})
                self.cov[1].update({key: dumI})
#----------------------------------------------------------------------------------------------



###########################################
###########################################
########## NEWEST PITD STUFF ##############
###########################################
###########################################
class pitd(common_pseudoDist):
    def __init__(self,h5,matelems,cfgs,gam,tmin,tmax,applyCP=False):
        self.h=h5py.File(h5,'r')
        self.mats=matelems
        self.cfgs=cfgs
        self.gam=gam
        self.tmin=int(tmin)
        self.tmax=int(tmax)
        self.amplitudes={a: {} for a in list(self.h.keys())}
        self.reducedAmps={a: {} for a in list(self.h.keys())}
        self.applyCP=applyCP


    #--------------------------------------------------------------------------------------------
    def reducedPITD(self,amp,renormWithMatelems=False,nuOrZSqrEqualZeroAmp=None):
        '''
        nuOrZSqrEqualZeroAmp sets a distinct amplitude other than 'amp' to form the rpITD double ratio
        
        For example, with nuOrZSqrEqualZeroAmp(\equiv A) != None
             rpitd(amp)[nu,z^2] = (amp[nu,z^2]/A[nu,0]) / (A[0,z^2]/A[0,0])
        '''
        altAmp = amp
        if nuOrZSqrEqualZeroAmp != None:
            altAmp = nuOrZSqrEqualZeroAmp
            
        
        for K in self.amplitudes[amp].keys():
            
            thisPf, thisPi = K.split(',')[0].split('_'); thisZ=K.split(',')[1]
            tuplePf=as_tuple(thisPf); tuplePi=as_tuple(thisPi); tupleZ=as_tuple(thisZ)

            # Initialize keys
            keyBnd, keyN, keyNnd = None, None, None
            bKeys=[]; bndKeys=[]; nKeys=[]; nndKeys=[]

            #-----------------------
            # Start by setting keys
            #-----------------------
            if not self.applyCP:
                keyBnd=pitdKey(thisPf,thisPi,"zsep000")
                keyN=pitdKey("pf000","pi000",thisZ)
                keyNnd=pitdKey("pf000","pi000","zsep000")

                # Update the reducedAmps dict
                self.reducedAmps[amp].update({K: {'real': {'avg': 0.0, 'err': 0.0, 'jks': []}, 'imag': {'avg': 0.0, 'err': 0.0, 'jks': []}}})
            else:
                # Only want positive momenta and displacements when applying CP
                if tuplePf[-1] < 0 or tupleZ[-1] < 0:
                    continue
                else:

                    masterKey=pitdKey(thisPf,thisPi,thisZ)
                    negP, negZ = tuple([-1*i for i in tuplePf]), tuple([-1*i for i in tupleZ])

                    # Update the reducedAmps dict
                    self.reducedAmps[amp].update({masterKey.key(): {'real': {'avg': 0.0, 'err': 0.0, 'jks': []}, 'imag': {'avg': 0.0, 'err': 0.0, 'jks': []}}})
                    
                    
                    bKeys   = [masterKey,pitdKey("pf%s%s%s"%(negP[0],negP[1],negP[2]),"pi%s%s%s"%(negP[0],negP[1],negP[2]),thisZ),\
                               pitdKey(thisPf,thisPi,"zsep%s%s%s"%(negZ[0],negZ[1],negZ[2])),\
                               pitdKey("pf%s%s%s"%(negP[0],negP[1],negP[2]),"pi%s%s%s"%(negP[0],negP[1],negP[2]),"zsep%s%s%s"%(negZ[0],negZ[1],negZ[2]))]
                    bndKeys = [pitdKey(thisPf,thisPi,"zsep000"),pitdKey("pf%s%s%s"%(negP[0],negP[1],negP[2]),"pi%s%s%s"%(negP[0],negP[1],negP[2]),"zsep000")]
                    nKeys   = [pitdKey("pf000","pi000",thisZ),pitdKey("pf000","pi000","zsep%s%s%s"%(negZ[0],negZ[1],negZ[2]))]
                    nndKeys = [pitdKey("pf000","pi000","zsep000")]
            #------------------------------------------------------------------------------------------------------------------------------
            # Done setting keys
            #------------------------------------------------------------------------------------------------------------------------------
                    


            B, Bnd, N, Nnd = None, None, None, None


            
            # If we are using bare matrix elements for renormalization, access all of them first...
            if renormWithMatelems:
                MATS_allZ = { keyBnd.key(): None, keyN.key(): None, keyNnd.key(): None }
                for matKey in MATS_allZ.keys():
                    sign=1.0
                    _pf, _pi = matKey.split(',')[0].split('_'); _z = matKey.split(',')[1]
                    if _pf == "pf000": sign=-1.0
                    
                    for rf in [1,2]:
                        for ri in [1,2]:
                            if rf == ri and _pf != "pf000": # cross-rows contribute in motion
                                continue
                            elif rf != ri and _pf == "pf000": # diag-rows contribute at rest
                                continue
                            else:
                                MAT_R, MAT_I = fit_util.SR(self.mats,self.cfgs,_pf,_pi,rf,ri,_z,self.gam,'real',\
                                                           self.tmin,self.tmax,pdfH5=False), \
                                                           fit_util.SR(self.mats,self.cfgs,thisPf,thisPi,rf,ri,thisZ,self.gam,'imag',\
                                                                       self.tmin,self.tmax,pdfH5=False)

                                # Only need fitted matrix element 'b'
                                for m in [MAT_R, MAT_I]: m.parse(parseOnly='b')
                                
                                tmp=[complex(_r,_i) for _r,_i in zip(MAT_R.params['b'],MAT_I.params['b'])]
                                if MATS_allZ[matKey]==None:
                                    MATS_allZ[matKey]=tmp
                                else:
                                    MATS_allZ[matKey]=[0.5*(rc1+sign*rc2) for rc1,rc2 in zip(MATS_allZ[matKey],tmp)]


                    # print(sum(MATS_allZ[matKey][:])/len(MATS_allZ[matKey]))

                    
                if not self.applyCP:
                    B=[complex(_r,_i) for _r,_i in zip(self.amplitudes[amp][K]['real']['jks'],self.amplitudes[amp][K]['imag']['jks'])]
                    Bnd=[v for v in MATS_allZ[keyBnd.key()]]
                    N=[v for v in MATS_allZ[keyN.key()]]
                    Nnd=[v for v in MATS_allZ[keyNnd.key()]]
                else:
                    print("Don't know what to do here...aborting...")
                    sys.exit()




            # For when not using bare matrix elements for ratio
            else:
                if not self.applyCP:
                    B=[complex(_r,_i) for _r,_i in zip(self.amplitudes[amp][K]['real']['jks'],self.amplitudes[amp][K]['imag']['jks'])]
                    Bnd=[complex(_r,_i) for _r,_i in zip(self.amplitudes[altAmp][keyBnd.key()]['real']['jks'],self.amplitudes[altAmp][keyBnd.key()]['imag']['jks'])]
                    N=[complex(_r,_i) for _r,_i in zip(self.amplitudes[altAmp][keyN.key()]['real']['jks'],self.amplitudes[altAmp][keyN.key()]['imag']['jks'])]
                    Nnd=[complex(_r,_i) for _r,_i in zip(self.amplitudes[altAmp][keyNnd.key()]['real']['jks'],self.amplitudes[altAmp][keyNnd.key()]['imag']['jks'])]
                else:
                    B=[(1.0/len(bKeys))*(complex(_r1,_i1)+complex(_r2,_i2).conjugate()+complex(_r3,_i3).conjugate()+complex(_r4,_i4))\
                       for _r1,_i1,_r2,_i2,_r3,_i3,_r4,_i4 in zip(self.amplitudes[amp][bKeys[0].key()]['real']['jks'],self.amplitudes[amp][bKeys[0].key()]['imag']['jks'],\
                                                                  self.amplitudes[amp][bKeys[1].key()]['real']['jks'],self.amplitudes[amp][bKeys[1].key()]['imag']['jks'],\
                                                                  self.amplitudes[amp][bKeys[2].key()]['real']['jks'],self.amplitudes[amp][bKeys[2].key()]['imag']['jks'],\
                                                                  self.amplitudes[amp][bKeys[3].key()]['real']['jks'],self.amplitudes[amp][bKeys[3].key()]['imag']['jks'])]

                    Bnd=[(1.0/len(bndKeys))*(complex(_r1,_i1)+complex(_r2,_i2))\
                         for _r1,_i1,_r2,_i2 in zip(self.amplitudes[altAmp][bndKeys[0].key()]['real']['jks'],self.amplitudes[altAmp][bndKeys[0].key()]['imag']['jks'],\
                                                    self.amplitudes[altAmp][bndKeys[1].key()]['real']['jks'],self.amplitudes[altAmp][bndKeys[1].key()]['imag']['jks'])]

                    N=[(1.0/len(nKeys))*(complex(_r1,_i1)+complex(_r2,_i2))\
                       for _r1,_i1,_r2,_i2 in zip(self.amplitudes[altAmp][nKeys[0].key()]['real']['jks'],self.amplitudes[altAmp][nKeys[0].key()]['imag']['jks'],\
                                                  self.amplitudes[altAmp][nKeys[1].key()]['real']['jks'],self.amplitudes[altAmp][nKeys[1].key()]['imag']['jks'])]

                    Nnd=[complex(_r,_i) for _r,_i in zip(self.amplitudes[altAmp][nndKeys[0].key()]['real']['jks'],self.amplitudes[altAmp][nndKeys[0].key()]['imag']['jks'])]


            # Now mess w/ ratio
            print("Start ratio")
            ratio=[(a/b)/(c/d) for a,b,c,d in zip(B,Bnd,N,Nnd)]
            print("End ratio")


            print("--> Done forming ratio for Amp,Key = %s, %s"%(amp,K))
            self.reducedAmps[amp][K]['real']['jks']=[v.real for v in ratio]
            self.reducedAmps[amp][K]['imag']['jks']=[v.imag for v in ratio]
            # sys.exit()


# COMMENT...
            # B, Bnd, N, Nnd = 0.0, 0.0, 0.0, 0.0
            # for j in range(0,self.cfgs):
            #     # Use original matrix elements to form reduced distributions
            #     if renormWithMatelems:
                   
            #         if not self.applyCP:
            #             B=complex(self.amplitudes[amp][K]['real']['jks'][j],self.amplitudes[amp][K]['imag']['jks'][j])
            #             Bnd=MATS_allZ[keyBnd.key()][j]
            #             N=MATS_allZ[keyN.key()][j]
            #             Nnd=MATS_allZ[keyNnd.key()][j]


            #     else:    
            #         if not self.applyCP:
            #             B=complex(self.amplitudes[amp][K]['real']['jks'][j],self.amplitudes[amp][K]['imag']['jks'][j])
            #             Bnd=complex(self.amplitudes[altAmp][keyBnd.key()]['real']['jks'][j],self.amplitudes[altAmp][keyBnd.key()]['imag']['jks'][j])
            #             N=complex(self.amplitudes[altAmp][keyN.key()]['real']['jks'][j],self.amplitudes[altAmp][keyN.key()]['imag']['jks'][j])
            #             Nnd=complex(self.amplitudes[altAmp][keyNnd.key()]['real']['jks'][j],self.amplitudes[altAmp][keyNnd.key()]['imag']['jks'][j])
            #         else:
            #             B=(1.0/len(bKeys))\
            #                 *(complex(self.amplitudes[amp][bKeys[0].key()]['real']['jks'][j],self.amplitudes[amp][bKeys[0].key()]['imag']['jks'][j])\
            #                   +complex(self.amplitudes[amp][bKeys[1].key()]['real']['jks'][j],self.amplitudes[amp][bKeys[1].key()]['imag']['jks'][j]).conjugate()\
            #                   +complex(self.amplitudes[amp][bKeys[2].key()]['real']['jks'][j],self.amplitudes[amp][bKeys[2].key()]['imag']['jks'][j]).conjugate()\
            #                   +complex(self.amplitudes[amp][bKeys[3].key()]['real']['jks'][j],self.amplitudes[amp][bKeys[3].key()]['imag']['jks'][j]))
                        
            #             Bnd=(1.0/len(bndKeys))\
            #                 *(complex(self.amplitudes[altAmp][bndKeys[0].key()]['real']['jks'][j],self.amplitudes[altAmp][bndKeys[0].key()]['imag']['jks'][j])\
            #                   +complex(self.amplitudes[altAmp][bndKeys[1].key()]['real']['jks'][j],self.amplitudes[altAmp][bndKeys[1].key()]['imag']['jks'][j]))
                        
            #             N=(1.0/len(nKeys))\
            #                 *(complex(self.amplitudes[altAmp][nKeys[0].key()]['real']['jks'][j],self.amplitudes[altAmp][nKeys[0].key()]['imag']['jks'][j])\
            #                   +complex(self.amplitudes[altAmp][nKeys[1].key()]['real']['jks'][j],self.amplitudes[altAmp][nKeys[1].key()]['imag']['jks'][j]))
                        
            #             Nnd=complex(self.amplitudes[altAmp][nndKeys[0].key()]['real']['jks'][j],self.amplitudes[altAmp][nndKeys[0].key()]['imag']['jks'][j])
                        
                        
                    
            #     ratio=complex(0,0)



            #     # # if Bnd != 0 and Nnd != 0:
            #     # try:
            #     #     ratio=(B/Bnd)/(N/Nnd)
            #     # except:
            #     #     try:
            #     #         ratio=B/N
            #     #     except:
            #     #         ratio=0

            #     ratio=(B/Bnd)/(N/Nnd)

            #     if amp == 'R':
            #         # ratio=B/N
            #         if tupleZ[-1] != 0:
            #             ratio/=(1.0*tupleZ[-1]**2)


            #     if amp == 'R':
            #         print("IN rpitd RATIO = [%.5f,%.5f]"%(ratio.real,ratio.imag))
                    
                    
            #     self.reducedAmps[amp][K]['real']['jks'].append(ratio.real)
            #     self.reducedAmps[amp][K]['imag']['jks'].append(ratio.imag)
# UNCOMMENT...


            # Now determine the average of each component of this reducedAmps key
            print("    Determing the average of each component of this reducedAmps key")
            for comp in ['real', 'imag']:
                self.reducedAmps[amp][K][comp]['avg']=avg(self.reducedAmps[amp][K][comp]['jks'])

                for j in range(0,self.cfgs):
                    self.reducedAmps[amp][K][comp]['err']+=np.power(self.reducedAmps[amp][K][comp]['jks'][j]-self.reducedAmps[amp][K][comp]['avg'],2)

                # self.reducedAmps[amp][K][comp]['err']=sum(np.power(self.reducedAmps[amp][K][comp]['jks'][:]-self.reducedAmps[amp][K][comp]['avg'],2))
                self.reducedAmps[amp][K][comp]['err']=np.sqrt( ((self.cfgs-1)/(1.0*self.cfgs))*self.reducedAmps[amp][K][comp]['err'] )

            print("    Avg/err of this reducedAmps key = Re [ %.5f, %.5f ], Im [ %.5f, %.5f ]"\
                  %(self.reducedAmps[amp][K]['real']['avg'],\
                    self.reducedAmps[amp][K]['real']['err'],\
                    self.reducedAmps[amp][K]['imag']['avg'],\
                    self.reducedAmps[amp][K]['imag']['err']))
                        
                             

            



###########################################
###########################################
###### GENERALIZED PITD STUFF #############
###########################################
###########################################
class pgitd(common_pseudoDist):
    def __init__(self,h5,cfgs,gam,tmin,tmax,pf=None,pi=None,z=None):
        self.h=h5py.File(h5,'r')
        self.cfgs=cfgs
        self.pf=pf
        self.pi=pi
        self.z=z
        self.gam=gam
        self.tmin=int(tmin)
        self.tmax=int(tmax)
        self.amplitudes={a: {} for a in list(self.h.keys())}
        self.reducedAmps={a: {} for a in list(self.h.keys())}
        self.rDist = { 1: {'jks': [], 'avg': 0.0, 'err': 0.0 },
                       2: {'jks': [], 'avg': 0.0, 'err': 0.0 } }

    def reducedPGITD(self,pitd,doubleRatio=True,thisAmpInDblRatio='A1'):
        '''
        Options:
        doubleRatio: use double ratio in forming reduced distribution

        thisAmpInDblRatio: if doubleRatio is true, then select this amplitude in double ratio ratio for amplitude Ai ==>  R = Ai(nu,xi,t,z^2)/thisAmpInDblRatio(nu,xi,t,0) x [M(0,0,0,z^2)/M(0,0,0,0)]
        '''
        pitdData = pitd

        # Run over each amplitude label in self.amplitudes and collect all (pf,pi,zsep) keys
        for amp,ampKeys in self.amplitudes.items():

            # Loop over each (pf,pi,zsep) key and make reduced distribution
            for ampKey in ampKeys:
                print("Forming reduced pGITD %s for (pf,pi,z) = %s"%(amp,ampKey))
                self.reducedAmps[amp].update({ampKey: {'real': {'avg': 0.0, 'err': 0.0, 'jks': generic1D(dat=np.zeros(self.cfgs))},\
                                                             'imag': {'avg': 0.0, 'err': 0.0, 'jks': generic1D(dat=np.zeros(self.cfgs))}} })

                # Convenience
                v = self.reducedAmps[amp][ampKey]
                

                keyBnd=pitdKey(ampKey.split(',')[0].split('_')[0],\
                               ampKey.split(',')[0].split('_')[1],"zsep000")
                keyN=pitdKey("pf000","pi000",ampKey.split(',')[1])
                keyNnd=pitdKey("pf000","pi000","zsep000")


                
                B, Bnd, N, Nnd = None, None, None, None


                # Proceed based on what datatype pitd is
                if type(pitdData) is not dict:
                    B=[complex(_r,_i) for _r,_i in zip(self.amplitudes[amp][ampKey]['real']['jks'],self.amplitudes[amp][ampKey]['imag']['jks'])]
                    Bnd=[complex(_r,_i) for _r,_i in zip(self.amplitudes[thisAmpInDblRatio][keyBnd.key()]['real']['jks'],self.amplitudes[thisAmpInDblRatio][keyBnd.key()]['imag']['jks'])]
                    N=[complex(_r,_i) for _r,_i in zip(pitdData.amplitudes['Y'][keyN.key()]['real']['jks'],pitdData.amplitudes['Y'][keyN.key()]['imag']['jks'])]
                    Nnd=[complex(_r,_i) for _r,_i in zip(pitdData.amplitudes['Y'][keyNnd.key()]['real']['jks'],pitdData.amplitudes['Y'][keyNnd.key()]['imag']['jks'])]
                

                elif type(pitdData) is dict:
                    absDisp='' # Needed, since pitd fits were done for averaged +/-z
                    for ele in as_tuple(ampKey.split(',')[1]):
                        absDisp+=str(abs(ele))
                    # Access appropriate pitd data
                    N_r = generic1D(pitdData[absDisp].amplitude[1].params['b'])
                    N_i = generic1D(pitdData[absDisp].amplitude[2].params['b'])
                    Nnd_r = generic1D(pitdData["000"].amplitude[1].params['b'])
                    Nnd_i = generic1D(pitdData["000"].amplitude[2].params['b'])
                    

                    B=[complex(_r,_i) for _r,_i in zip(self.amplitudes[amp][ampKey]['real']['jks'],self.amplitudes[amp][ampKey]['imag']['jks'])]
                    Bnd=[complex(_r,_i) for _r,_i in zip(self.amplitudes[thisAmpInDblRatio][keyBnd.key()]['real']['jks'],self.amplitudes[thisAmpInDblRatio][keyBnd.key()]['imag']['jks'])]
                    N=[complex(_r,_i) for _r,_i in zip(N_r.dat,N_i.dat)]
                    Nnd=[complex(_r,_i) for _r,_i in zip(Nnd_r.dat,Nnd_i.dat)]
                    # for j in range(0,self.cfgs):
                    #     # The pGITD data for some zsep and zsep=000 as complexes
                    #     B=complex(self.amplitudes[amp][ampKey]['real']['jks'][j],self.amplitudes[amp][ampKey]['imag']['jks'][j])
                    #     Bnd=complex(self.amplitudes[amp][keyBnd.key()]['real']['jks'][j],self.amplitudes[amp][keyBnd.key()]['imag']['jks'][j])
                        
                        
                    #     # The pITD data for some abs(zsep) and zsep=000 as complexes
                    #     N=complex(N_r.dat[j],N_i.dat[j])
                    #     Nnd=complex(Nnd_r.dat[j],Nnd_i.dat[j])
                        
                        
                    #     ratio=complex(0.0)
                    #     try:
                    #         # ratio=(B/Bnd)/(N/Nnd)
                    #         ratio=B/N
                    #     except:
                    #         print("Got something non-finite, defaulting to 100")
                    #         ratio=100
                            
                            
                    #     v['real']['jks'].dat[j]=ratio.real
                    #     v['imag']['jks'].dat[j]=ratio.imag
                        

                    # # Now determine the average of each component of this reducedAmps key
                    # for comp in ['real', 'imag']:
                    #     v[comp]['jks'].average(); v[comp]['avg'] = v[comp]['jks'].avg
                        
                    #     for j in range(0,self.cfgs):
                    #         v[comp]['err']+=np.power(v[comp]['jks'].dat[j]-v[comp]['avg'],2)
                    #     v[comp]['err']=np.sqrt( ((self.cfgs-1)/(1.0*self.cfgs))*v[comp]['err'] )



                # Now mess w/ ratio
                print("Start ratio")
                try:
                    ratio=[(a/b)/(c/d) if doubleRatio else a/c for a,b,c,d in zip(B,Bnd,N,Nnd)]
                    # ratio=[(a/b) for a,b in zip(B,Bnd)]
                except:
                    try:
                        ratio=[a/c for a,c in zip(B,N)]
                    except:
                        ratio=[-1 for i in range(0,len(B))]
                print("End ratio")
                # print("Ratio = %.5f"%(sum(ratio)/len(ratio)))
                print(ratio)

                print("--> Done forming ratio for Amp,Key = %s, %s"%(amp,ampKey))
                self.reducedAmps[amp][ampKey]['real']['jks']=[v.real for v in ratio]
                self.reducedAmps[amp][ampKey]['imag']['jks']=[v.imag for v in ratio]


                # Now determine the average of each component of this reducedAmps key
                print("    Determining the average of each component of this reducedAmps key")
                for comp in ['real', 'imag']:
                    self.reducedAmps[amp][ampKey][comp]['avg']=avg(self.reducedAmps[amp][ampKey][comp]['jks'])
                    
                    for j in range(0,self.cfgs):
                        self.reducedAmps[amp][ampKey][comp]['err']+=np.power(self.reducedAmps[amp][ampKey][comp]['jks'][j]-self.reducedAmps[amp][ampKey][comp]['avg'],2)
                        
                    self.reducedAmps[amp][ampKey][comp]['err']=np.sqrt( ((self.cfgs-1)/(1.0*self.cfgs))*self.reducedAmps[amp][ampKey][comp]['err'] )
                    
                    # print("    Avg/err of this reducedAmps key = Re [ %.5f, %.5f ], Im [ %.5f, %.5f ]"\
                    #       %(self.reducedAmps[amp][K]['real']['avg'],\
                    #         self.reducedAmps[amp][K]['real']['err'],\
                    #         self.reducedAmps[amp][K]['imag']['avg'],\
                    #         self.reducedAmps[amp][K]['imag']['err']))
