#This script is based on Main01.py 

# main01.py is a part of the PYTHIA event generator.
# Copyright (C) 2016 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
#
# This is a simple test program. It fits on one slide in a talk.  It
# studies the charged multiplicity distribution at the LHC. To set the
# path to the Pythia 8 Python interface do either (in a shell prompt):
#      export PYTHONPATH=$(PREFIX_LIB):$PYTHONPATH
# or the following which sets the path from within Python.

##----------------------------------------------------------------------------------------------------------------------
# LINKING PYTHON WITH PYTHIA 

# ./configure --with-python-include=/cms/base/cmssoft/slc6_amd64_gcc530/external/python/2.7.11-giojec4/include/python2.7/ --with-python-bin=/cms/base/cmssoft/slc6_amd64_gcc530/cms/cmssw/CMSSW_8_1_0/external/slc6_amd64_gcc530/bin/ --with-hepmc2=/het/p1/mbuckley/HepMC/install/

##----------------------------------------------------------------------------------------------------------------------
#TO RUN THE SCRIPT

# python Seb_LHE_image.py LHE_FILE Nevents jets_filename subjets_filename 
#Example:  nohup  python Seb_LHE_image.py tt_1000_unweighted.lhe 10 jets.npy subjets.npy &
#Where Nevents are the number of events of the LHE file and jets_filename and subjets_filename the name of the files where we save the numpy arrays of jet/subjet (pT,eta,phi)

##----------------------------------------------------------------------------------------------------------------------

import sys
import os
cfg = open("Makefile.inc")
lib = "../lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Import the Pythia module.
import pythia8

full_event_plots='tt_plots'
os.system('mkdir -p '+full_event_plots)

Nevents=int(sys.argv[2])
jets_filename=sys.argv[3]
subjets_filename=sys.argv[4]


# Import SlowJet: we use SlowJet to cluster the jet constituents. SlowJet is now identical to FastJets but has less features.
#The recent introduction of fjcore, containing the core functionality of FastJet in a very much smaller package, has changed the conditions. It now is possible (even encouraged by the authors) to distribute the two fjcore files as part of the PYTHIA package. Therefore the SlowJet class doubles as a convenient front end to fjcore, managing the conversion back and forth between PYTHIA and FastJet variables. Some applications may still benefit from using the native codes, but the default now is to let SlowJet call on fjcore for the jet finding. More info at: http://home.thep.lu.se/~torbjorn/pythia82html/Welcome.html
pythia = pythia8.Pythia()

##----------------------------------------------------------------------------------------------------------------------
## READ LHE FILE
if len(sys.argv) > 1: lhe_file_name = sys.argv[1]

pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = "+lhe_file_name)

pythia.init() #Initializes the process. Incoming p p beams are the default

mult = pythia8.Hist("charged multiplicity", 100, -0.5, 799.5)

#Common parameters for the 2 jet finder:
p_value=-1 #This dettermines the clustering algorithm. p = -1 corresponds to the anti-kT one, p = 0 to the Cambridge/Aachen one and p = 1 to the kT one.
eta_max=float(sys.argv[5])
R=float(sys.argv[6])
pTjetMin=int(sys.argv[7]) #Min pT for all jets
# eta_max=2.5
# R=1.5
# pTjetMin=300 #Min pT for all jets
#Exclude neutrinos (and other invisible) from study:
nSel=2

# calorimeter granularity
etaedges=np.arange(-eta_max,eta_max+0.025,10/206.25)
phiedges=np.arange(-np.pi,np.pi+0.024,np.pi/130.)
# print etaedges, phiedges
my_map=plt.cm.gray

#Set up SlowJet jet finder, with anti-kT clustering and pion mass assumed for non-photons..
slowJet = pythia8.SlowJet(p_value, R, pTjetMin, eta_max, nSel, 1)
jet_list=[]
jet_parton_list=[]
jet_mass=[]

##----------------------------------------------------------------------------------------------------------------------
# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0, Nevents):
    if not pythia.next(): continue

   # for i in range(0,pythia.event.size()):
   #     print('i {},id {}'.format(i,pythia.event[i].id()))

    # Find number of all final charged particles and fill histogram.
    nCharged = 0
    for prt in pythia.event:
        if prt.isFinal() and prt.isCharged(): nCharged += 1
    mult.fill(nCharged)

    ##---------------------------------------------------------------------------------------------------------------
    # Analyze Slowet jet properties. List first few JETS.
    slowJet.analyze(pythia.event) #To analyze the event with Slowjet
    if (iEvent < Nevents):
        slowJet.list() #To list the jets
    
    ##---------------------------------------------------------------------------------------------------------------
    # dump jet info into 4-vectors: pT,eta,Phi,E
    jet_list.append([ ])# [] for i in range(slowJet.sizeJet())])
    jet_parton_list.append([ ])# [] for i in range(slowJet.sizeJet())])
    jet_mass.append([])
    
   

    for j in range(0,slowJet.sizeJet()):
        #vec=slowJet.p(j)
        jet_list[iEvent].append(slowJet.p(j))# Gives the jets 4-vector: px,py,pz,E
#        if (iEvent < Nevents):
#            print('Event {} ,contituents{}'.format(iEvent, slowJet.constituents(j)))

        jet_parton_list[iEvent].append([pythia.event[c].p() for c in slowJet.constituents(j)])# We read the jet constituents. Gives the 4-vector: px,py,pz,E. We can the access with pT(), eta(), Phi()
        jet_mass[iEvent].append(slowJet.m(j)) #We get a list of the mass of the leading jet in pT for each event
    #print('Jet 0 mass = {}'.format(jet_mass))
       
##---------------------------------------------------------------------------------------------------------------------
# End of event loop. Statistics. Histogram. Done.
pythia.stat();
#print(mult)

njets_vec = map(len,jet_list)
# print jet_parton_list
# plotting event N
bins = 100

const_pT, const_eta, const_phi, jetcircle = [], [], [], []
subjet_pT,subjet_eta,subjet_phi = [],[],[]
subjet2_pT=[]
jet_pT,jet_eta,jet_phi,jet_Mass = [],[],[],[]

##----------------------------------------------------------------------------------------------------------------------
# plotting constituents of all jets in all events
for NeventPlot in range(0,Nevents):
    #Njet = range(len(jet_list[NeventPlot]))
    const_pT.append([ ])
    const_eta.append([ ])
    const_phi.append([ ])
    jetcircle.append([ ])

#Jets are sorted from Greater to lower pT. As we want to keep the hardest jet, we then just take jet_list[NeventPlot][0]
    if len(jet_list[NeventPlot])>0:
        ijet=0
        #Create jet constituents pT,eta,phi lists
        const_pT[NeventPlot].append([jet.pT() for jet in jet_parton_list[NeventPlot][ijet]])
        const_eta[NeventPlot].append([jet.eta() for jet in jet_parton_list[NeventPlot][ijet]])
        const_phi[NeventPlot].append([jet.phi() for jet in jet_parton_list[NeventPlot][ijet]])

        subjet_pT.append([jet.pT() for jet in jet_parton_list[NeventPlot][ijet]])
        #subjet2_pT.append(jet_parton_list[NeventPlot][ijet].pT())
        subjet_eta.append([jet.eta() for jet in jet_parton_list[NeventPlot][ijet]])
        subjet_phi.append([jet.phi() for jet in jet_parton_list[NeventPlot][ijet]])

        #jet pT,eta,phi
        jet_pT.append(jet_list[NeventPlot][ijet].pT())
        jet_eta.append(jet_list[NeventPlot][ijet].eta())
        jet_phi.append(jet_list[NeventPlot][ijet].phi())

        jet_Mass.append(jet_mass[NeventPlot][ijet])
        #jet_mass.append([])
        #jet_mass[NeventPlot].append(slowJet.m(ijet))
        #print('Jet 0 mass = {}'.format(jet_Mass))

        jetcircle[NeventPlot].append(plt.Circle((jet_list[NeventPlot][ijet].eta(), jet_list[NeventPlot][ijet].phi()), radius=R, color='r', fill=False))


        #for ijet in range(len(jet_list[NeventPlot])):
        #        print('One event pT constituents for jet {}:--- {} '.format(ijet,[jet.pT() for jet in jet_parton_list[NeventPlot][ijet]]))
        #        print('-------------'*10)

    
        all_jet_pT = np.concatenate(const_pT[NeventPlot]) #concatenate joins all vectors into 1
        #print('all_jet_pT {}'.format(all_jet_pT))
        #print('const_pT {}'.format(const_pT[NeventPlot]))
        #print('----------------'*15)
        all_jet_eta = np.concatenate(const_eta[NeventPlot])
        all_jet_phi = np.concatenate(const_phi[NeventPlot])


#    print('One event Jets constituents PT {}'.format(all_jet_pT))
#    print('-------------'*10)
#    print('One event Jets constituents  eta {}'.format(all_jet_eta))
#    print('-------------'*10)
#    print('One event Jets constituents phi {}'.format(all_jet_phi))

        ##---------------------------------------------------------------------------------------------------------------------# -
#         #Create 2-D histogram
#         jets_h, xedges, yedges = np.histogram2d(all_jet_eta,all_jet_phi,bins=(etaedges,phiedges),weights=all_jet_pT)
# 
#         # print jet_h
#         # print xedges, yedges         
#         fig = plt.gcf()
#         ax = fig.gca()
#         #We add circles center at each jet
#         for crc in jetcircle[NeventPlot]:
#             ax.add_artist(crc)
# 
#         #plt.pcolor(xedges, yedges,np.swapaxes(jets_h,0,1),cmap=my_map)
#         plt.pcolor(xedges, yedges,np.swapaxes(jets_h,0,1))
#         plt.xlim(-eta_max,eta_max)
#         plt.xlabel('eta')
#         plt.ylim(-np.pi,np.pi)
#         plt.ylabel('phi')
#         #ax = plt.add_subplot(111)
#         # plt.show()
#         plt.savefig(full_event_plots+'/hist_'+str(NeventPlot)+'.png')
#         fig.clf() #Clears the figure to start the next event in the loop. (If not I would have multiples circles from previous events in the plots

##-----------------------------------------------------------------------------------------------------------------------------
#We create jet list and constituents list with (pT,eta,phi) for all events to write in an output file

subjet_pTetaPhi=(subjet_pT,subjet_eta,subjet_phi) #format: ([[[pT_jet1_constituents, pT_jet2_constituents,...]]],[[[eta_jet1_constituents, eta_jet2_constituents,...]]],[[[phi_jet1_constituents, phi_jet2_constituents,...]]])
#print('subjet_pTetaPhi: {}'.format(subjet_pTetaPhi))

jet_pTetaPhi=(jet_pT,jet_eta,jet_phi,jet_Mass) #format: ([[[pT_jet1_constituents, pT_jet2_constituents,...]]],[[[eta_jet1_constituents, eta_jet2_constituents,...]]],[[[phi_jet1_constituents, phi_jet2_constituents,...]]])
#print('jet_pTetaPhi: {}'.format(jet_pTetaPhi[3]))

##----------------------------------------------------------------------------------------------------------------------
#We save the jet and constituents list as .npy output files

#from tempfile import TemporaryFile
#np_jets = TemporaryFile()
np.save(jets_filename+'.npy',jet_pTetaPhi)
np.save(subjets_filename+'.npy',subjet_pTetaPhi)

#print('subjet pT{}'.format(subjet_pT))
#print('subjet2 pT {}'.format(subjet2_pT))

