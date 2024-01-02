#!/usr/bin/python3.8

# Collect all pheno PDF stuff in one location

import numpy as np
from matplotlib import cm
import lhapdf
lhapdf.pathsPrepend("/home/colin/LHAPDF-6.3.0/share/LHAPDF/LHAPDF")

cmap_autumn = cm.get_cmap('autumn')
cmap_winter = cm.get_cmap('winter')
cmap_cool = cm.get_cmap('cool')
cmap_Wistia = cm.get_cmap('Wistia')

class pheno:
    def __init__(self,phenoQ,dirac,darkBkgd,phenoAlpha=0.23,phenoHatch=''):
        self.phenoQ=phenoQ
        self.dirac=dirac
        self.darkBkgd=darkBkgd
        self.phenoAlpha=phenoAlpha
        self.phenoHatch=phenoHatch
        self.phenoX=np.linspace(0.0,1.0,500)
        self.phenoNU=np.linspace(0,20,300)

        # Handles to individual datasets
        self.F={}
        self.F_ITD={} # Master pheno ITD dictionary
        self.pdfSets=[]
        self.lhapdfPDFList=None
        self.phenoCols=[]


    # Get each set of pheno PDF desired - n.b. these are personally selected by hand; nothing fancy
    def accessPDFs(self):
        if self.dirac == 8:
            CJ15={}; MSTW={}; JAM20={}; NNPDF={}
            self.lhapdfPDFList=[CJ15, MSTW, JAM20, NNPDF]
            self.pdfSets=[("CJ15nlo",2,'CJ15'), ("MSTW2008nnlo68cl_nf4",3,'MSTW'),\
                          ("JAM20-SIDIS_PDF_proton_nlo",2,'JAM20'),\
                          ("NNPDF31_nnlo_pch_as_0118_mc_164",1,'NNPDF')]
            self.phenoCols=['black','goldenrod','darkgreen','darkcyan']
            if self.darkBkgd:
                self.phenoCols=[cmap_autumn(255),cmap_winter(255),cmap_winter(255),cmap_cool(255)]
            
        elif self.dirac == 11:
            NNPDF={}; JAM17={}; JAM22={}
            self.lhapdfPDFList=[NNPDF, JAM17, JAM22]
            self.pdfSets=[("NNPDFpol11_100",1,'NNPDFpol1.1'),\
                          ("JAM17_PPDF_nlo",1,'JAM17'),("JAM22-PPDF_proton_nlo",1,'JAM22')]
            self.phenoCols=['#cb4b16','#2aa198','#657b83']
            if self.darkBkgd:
                self.phenoCols=[cmap_autumn(20),cmap_Wistia(255),cmap_autumn(255)]            
        else:
            raise ValueError("No selected pheno PDFs")


        # For each member of lhapdfPDFList, init pdf dicts for all light quark combos
        for q in ['u', 'ubar', 'd', 'dbar', 'qv', 'q+', 'q', 'qbar']:
            for s in self.lhapdfPDFList:
                s.update({q: {'central': [], 'errplus': [], 'errminus': []}})

        # Actually access the lhapdf sets here
        for n,p in enumerate(self.pdfSets):
            self.F.update({lhapdf.getPDFSet(p[0]): {'pdf2parse': lhapdf.getPDFSet(p[0]).mkPDFs(),\
                                                    'pdf2plot': self.lhapdfPDFList[n],\
                                                    'color': self.phenoCols[n], 'label': p[2]}})


    # Parse each member of PDF set
    def lhaParse(self,xi,pdfSet,PDFs):
        uAll=[0.0 for i in range(pdfSet.size)]
        ubarAll=[0.0 for i in range(pdfSet.size)]
        dAll=[0.0 for i in range(pdfSet.size)]
        dbarAll=[0.0 for i in range(pdfSet.size)]
        qvAll=[0.0 for i in range(pdfSet.size)]
        qplusAll=[0.0 for i in range(pdfSet.size)]
        qAll=[0.0 for i in range(pdfSet.size)]
        qbarAll=[0.0 for i in range(pdfSet.size)]
        
        # Get value of all PDF members for passed value of x
        for mem in range(pdfSet.size):
            uAll[mem]=PDFs[mem].xfxQ(2,xi,self.phenoQ)*(1.0/xi)
            ubarAll[mem]=PDFs[mem].xfxQ(-2,xi,self.phenoQ)*(1.0/xi)
            dAll[mem]=PDFs[mem].xfxQ(1,xi,self.phenoQ)*(1.0/xi)
            dbarAll[mem]=PDFs[mem].xfxQ(-1,xi,self.phenoQ)*(1.0/xi)
            qvAll[mem]=uAll[mem]-ubarAll[mem]-(dAll[mem]-dbarAll[mem]) # 1/x managed just above
            qplusAll[mem]=uAll[mem]+ubarAll[mem]-(dAll[mem]+dbarAll[mem])
            qAll[mem]=uAll[mem]-dAll[mem] # 1/x managed just above
            qbarAll[mem]=ubarAll[mem]-dbarAll[mem] # 1/x managed just above

            
        # Determine the uncertainties
        unc={'u': None, 'ubar': None, 'd': None, 'dbar': None, 'qv': None, 'q+': None,\
             'q': None, 'qbar': None}
        unc.update({'u': pdfSet.uncertainty(uAll)})
        unc.update({'ubar': pdfSet.uncertainty(ubarAll)})
        unc.update({'d': pdfSet.uncertainty(dAll)})
        unc.update({'dbar': pdfSet.uncertainty(dbarAll)})
        unc.update({'qv': pdfSet.uncertainty(qvAll)})
        unc.update({'q+': pdfSet.uncertainty(qplusAll)})
        unc.update({'q': pdfSet.uncertainty(qAll)})
        unc.update({'qbar': pdfSet.uncertainty(qbarAll)})
        
        return unc


    # Extract pheno PDFs and errors
    def extractPDFs(self):
        for ni, xi in enumerate(self.phenoX):
            for pdfSet, pdfs in self.F.items():
                # Catch the uncertainty of this pheno member
                parse=self.lhaParse(xi,pdfSet,pdfs['pdf2parse'])
                for flavor in ['u', 'ubar', 'd', 'dbar', 'qv', 'q+', 'q', 'qbar']:
                    pdfs['pdf2plot'][flavor]['central'].append( parse[flavor].central )
                    pdfs['pdf2plot'][flavor]['errplus'].append( parse[flavor].errplus )
                    pdfs['pdf2plot'][flavor]['errminus'].append( parse[flavor].errminus )


    # Write out extracted pheno PDFs to text files
    def writePhenoPDFs(self):
        for pdfSet, pdfs in self.F.items():
            for flavor in ['u', 'ubar', 'd', 'dbar', 'qv', 'q+', 'q', 'qbar']:
                zippy = zip(self.phenoX, pdfs['pdf2plot'][flavor]['central'],\
                            pdfs['pdf2plot'][flavor]['errplus'],\
                            pdfs['pdf2plot'][flavor]['errminus'])
    
        np.savetxt('%s-PDF.%s.dat'%(pdfs['label'],flavor), zippy, fmt='%.7f %.7f %.7f %.7f')



    # Compute pheno ITDs
    def computePhenoITDs(self):
        if self.dirac == 8:
            CJ15_ITD={}; MSTW_ITD={}; JAM20_ITD={}; NNPDF_ITD={}
            for q in ['qv', 'q+', 'q', 'qbar']:
                for s in [CJ15_ITD, MSTW_ITD, JAM20_ITD, NNPDF_ITD]:
                    s.update({q: {'central': [], 'errplus': [], 'errminus': []}})

            self.F_ITD={setCJ15NLO: {'pdf2parse': cj15pdfs, 'itd2plot': CJ15_ITD,\
                                     'out': 'cj15nlo','color': self.F[setCJ15NLO]['color'],\
                                     'label': 'CJ15'},\
                        setMSTWNNLO: {'pdf2parse': mstwpdfs, 'itd2plot': MSTW_ITD,\
                                      'out': 'mstwnnlo','color': self.F[setMSTWNNLO]['color'],\
                                      'label': 'MSTW'},\
                        setJAM20NLO: {'pdf2parse': jam20pdfs, 'itd2plot': JAM20_ITD,\
                                      'out': 'jam20nlo','color': self.F[setJAM20NLO]['color'],\
                                      'label': 'JAM20'},\
                        setNNPDFNNLO: {'pdf2parse': nnpdfpdfs, 'itd2plot': NNPDF_ITD,\
                                       'out': 'nnpdfnnlo','color': self.F[setNNPDFNNLO]['color'],\
                                       'label': 'NNPDF'}}

        
            for mi, nu_i in enumerate(self.phenoNU):
                for pdfSet, pdfitd in self.F_ITD.items():
                    for flavor, fourier in [('qv',np.cos),('q+',np.sin)]: # q,qbar
                    
                        pdfitd['itd2plot'][flavor]['central'].\
                            append(integrate.quad(lambda xph : fourier(nu_i*xph)*\
                                                  lhaParse(xph,pdfSet,pdfitd['pdf2parse'])\
                                                  [flavor].central,0,1)[0])
                        pdfitd['itd2plot'][flavor]['errplus'].\
                            append(integrate.quad(lambda xph : fourier(nu_i*xph)*\
                                                  lhaParse(xph,pdfSet,pdfitd['pdf2parse'])\
                                                  [flavor].errplus,0,1)[0])
                        pdfitd['itd2plot'][flavor]['errminus'].\
                            append(integrate.quad(lambda xph : fourier(nu_i*xph)*\
                                                  lhaParse(xph,pdfSet,pdfitd['pdf2parse'])\
                                                  [flavor].errminus,0,1)[0])
        else:
            raise ValueError("Pheno ITD computation not set up yet for dirac = %i"%self.dirac)


    # Write out computed pheno ITDs and then exit
    def writePhenoITDs(self):
        for pdfSet, itd in self.F_ITD.items():
            for flavor in ['qv','q+']: # 'q', 'qbar']:
                zippy = zip(itd['itd2plot'][flavor]['central'],\
                            itd['itd2plot'][flavor]['errplus'],\
                            itd['itd2plot'][flavor]['errminus'])

                # n.b. 'list(zippy)' replaces just 'zippy' in python3 vs. python2
                np.savetxt('%s-ITD.%s.dat'%(itd['out'],flavor), list(zippy), fmt='%.7f %.7f %.7f')
            
        sys.exit()
