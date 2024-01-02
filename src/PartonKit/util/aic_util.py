#/usr/bin/python3

import math
import numpy as np
import pdf_utils

# #########################
# # PRIORS
# #########################
# locationParam={'\\alpha': -1, '\\beta': 0}
# mean={'\\alpha': 0.0, '\\beta': 3.0}
# mean.update({'C^{lt}_%i'%n: 0.0 for n in range(0,8)})
# mean.update({'C^{az}_%i'%n: 0.0 for n in range(0,8)})
# mean.update({'C^{t4}_%i'%n: 0.0 for n in range(0,8)})
# mean.update({'C^{t6}_%i'%n: 0.0 for n in range(0,8)})

# width={'\\alpha': 0.25, '\\beta': 0.5,
#        'C^{lt}_0': 1.1, 'C^{lt}_1': 0.75, 'C^{lt}_2': 0.5, 'C^{lt}_3': 0.25, 'C^{lt}_4': 0.125, 'C^{lt}_5': 0.1, 'C^{lt}_6': 0.05, 'C^{lt}_7': 0.025}
# for s in ['az','t4','t6']:
#     width.update({'C^{%s}_%i'%(s,n): 0.25 for n in range(0,3)})
#     width.update({'C^{%s}_%i'%(s,n): 0.125 for n in range(3,6)})
#     width.update({'C^{%s}_%i'%(s,n): 0.1 for n in range(6,8)})

    
# # # In actual minimization to obtain pdfs, widths were given as is
# # # ...............(width used) = 2*(normalized width)
# # # ...so divide all widths by 2 for proper normalization
# # for k in width.keys():
# #     width[k] = width[k]/2.0


# priors={'mean': mean, 'width': width}


#########################
# AIC CLASS
#########################
class AIC:
    def __init__(self,nDataFit=[],models=[],cuts=[]):
        self.x            = np.linspace(0,1,500)
        self.nDataFit     = nDataFit
        self.models       = models
        self.cuts         = cuts # [pmin,pmax,zmin,zmax]

        self.modelsAndCuts = dict(zip(self.models,self.cuts)) # pair cuts w/ models
        
        # LT AIC
        self.aicAvg       = np.zeros(len(self.x))
        self.aicErr       = np.zeros(len(self.x))
        # AZ AIC
        self.aicAvg_AZ       = np.zeros(len(self.x))
        self.aicErr_AZ       = np.zeros(len(self.x))
        # HT AIC
        self.aicAvg_HT       = np.zeros(len(self.x))
        self.aicErr_HT       = np.zeros(len(self.x))
        

        # Determine all AICc's
        self.aiccs = {m: self.aicN(self.models[i],self.nDataFit[i]) for i,m in enumerate(self.models)}
        # Minimum of all AICc's
        self.minAICcs = min(self.aiccs.values())
        self.minAICcs = 0.0 # 09/26/2022 maybe I don't need to subtract smallest aiccs value

        # Compute the sum of Boltzmann weights - \sum_i [ Exp(-Ai/2) ]
        self.sumAICcs = 0.0
        for i,m in enumerate(self.models):
            print("Current val sumAICcs = %.70f"%self.sumAICcs)
            self.sumAICcs += np.exp((-self.aiccs[m]-self.minAICcs)/2.0)

        # Compute all the weights
        self.aicWeights=[np.exp((-self.aiccs[M]-self.minAICcs)/2.0)/self.sumAICcs for M in self.models]


    def avgAIC(self):
        for n,M in enumerate(self.models):
            self.aicAvg[:]+=self.aicWeights[n]*M.pdfLT(self.x[:])

    def errAIC(self):
        for n, xi in enumerate(self.x):
            for m,M in enumerate(self.models):
                # self.aicErr[n] += self.aicWeights[m]*M.pdfLTErr(xi)[0]
                self.aicErr[n] += self.aicWeights[m]*(M.pdfLTErr(xi)[0]**2+(M.pdfLT(xi)-self.aicAvg[n])**2)

            self.aicErr[n] = np.sqrt(self.aicErr[n])


    def avgAIC_AZ(self):
        for n,M in enumerate(self.models):
            self.aicAvg_AZ[:]+=self.aicWeights[n]*M.pdfAZ(self.x[:])

    def errAIC_AZ(self):
        for n, xi in enumerate(self.x):
            for m,M in enumerate(self.models):
                self.aicErr_AZ[n] += self.aicWeights[m]*(M.pdfAZErr(xi)[0]**2+(M.pdfAZ(xi)-self.aicAvg_AZ[n])**2)
            self.aicErr_AZ[n] = np.sqrt(self.aicErr_AZ[n])


    def avgAIC_HT(self):
        for n,M in enumerate(self.models):
            for h in M.HT:
                self.aicAvg_HT[:]+=self.aicWeights[n]*h.pdf(self.x[:])

    def errAIC_HT(self):
        for n, xi in enumerate(self.x):
            for m,M in enumerate(self.models):
                for h in M.HT:
                    self.aicErr_HT[n] += self.aicWeights[m]*(h.pdfErr(xi)[0]**2+(h.pdf(xi)-self.aicAvg_HT[n])**2)
            self.aicErr_HT[n] = np.sqrt(self.aicErr_HT[n])

    
    # Return AICc for nth model
    def aicN(self,model,d):
        p=model.numParams
        try:
            aicc=2*model.pAvg['L2']+2*p+(2.0*p*(p+1))/(1.0*(d-p-1))
        except:
            print("Found aicN value that is infty (i.e. d-p-1 = 0), so this aicN will be set to 1e8")
            aicc=1e8
        return aicc


        
    # Write (x,avg AIC,err AIC) to file
    def out(self,out):
        with open(out, "w") as f:
            for n,x in enumerate(self.x):
                f.write("%.7f %.7f %.7f\n"%(x,self.aicAvg[n],self.aicErr[n]))
