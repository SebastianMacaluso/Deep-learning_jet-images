##=============================================================================================
##=============================================================================================
# CREATE AND PREPROCESS JET IMAGES TO BE USED AS THE INPUT OF A CONVOLUTIONAL NEURAL NETWORK
##=============================================================================================
##=============================================================================================

# This script loads .npy files as a list of numpy arrays ([[pT],[eta],[phi]]) and produces numpy arrays where each entry represents the intensity in transverse momentum (pT) for a pixel in a jet image. The script does the following:
# 1) We load .npy files with jets and jet constituents (subjets) lists of [[pT],[eta],[phi]]. We generate this files by running Pythia with SlowJets over an LHE file generated in Madgraph 5. 
# 2) We center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0).
# 3) We shift the coordinates of each jet constituent so that the jet is centered at the origin in the new coordinates.
# 4) We calculate the angle theta for the principal axis.
# 5) We rotate the coordinate system so that the principal axis is the same direction (+ eta) for all jets.
# 6) We scale the pixel intensities such that sum_{i,j} I_{i,j}=1
# 7) We create the array of pT for the jet constituents, where each entry represents a pixel. We add all the jet constituents that fall within the same pixel.
# 8) We reflect the image over the horizontal and vertical axes to ensure the 3rd maximum is on the upper right quarter-plane
# 9) We standardize the images adding a factor "bias" for noise suppression: Divide each pixel by the standard deviation of that pixel value among all the images in the training data set 
# 11) We output a tuple with the numpy arrays and true value of the images that we will use as input for our neural network
# 12) We plot all the images.
# 13) We add the images to get the average jet image for all the events.
# 14) We plot the averaged image.
# Last updated: October 10, 2017. Sebastian Macaluso
# Written for Python 3.6.0


#To run this script:
# python image_preprocess2.py signal_jets_subjets_directory background_jets_subjets_directory

#(To get the images from 09/13/2017)
# python image_preprocess_avgimg_presentation.py results_tt_200k_ptheavy800-900_pflow2 results_qcd_400k_ptj800-900_pflow2


##---------------------------------------------------------------------------------------------
#RESOLUTION of ECAL/HCAL ATLAS/CMS

# CMS ECal DeltaR=0.0175 and HCal DeltaR=0.0875 (https://link.springer.com/article/10.1007/s12043-007-0229-8 and https://cds.cern.ch/record/357153/files/CMS_HCAL_TDR.pdf ) 
# CMS: For the endcap region, the total number of depths is not as tightly constrained as in the barrel due to the decreased φ-segmentation from 5 degrees (0.087 rad) to 10 degrees for 1.74 < |η| < 3.0. (http://inspirehep.net/record/1193237/files/CMS-TDR-010.pdf)
#The endcap hadron calorimeter (HE) covers a rapidity region between 1.3 and 3.0 with good hermiticity, good
# transverse granularity, moderate energy resolution and a sufficient depth. A lateral granularity ( x ) was chosen
# 0.087 x 0.087. The hadron calorimeter granularity must match the EM granularity to simplify the trigger. (https://cds.cern.ch/record/357153/files/CMS_HCAL_TDR.pdf )

# ATLAS ECal DeltaR=0.025 and HCal DeltaR=0.1 (https://arxiv.org/pdf/hep-ph/9703204.pdf page 11)

##=============================================================================================
##=============================================================================================

##=============================================================================================
############       LOAD LIBRARIES
##=============================================================================================

import pickle
import gzip
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.set_printoptions(threshold=np.nan)

import scipy 

# from sklearn.preprocessing import scale
from sklearn import preprocessing

import h5py

import time
start_time = time.time()

##=============================================================================================
############       GLOBAL VARIABLES
##=============================================================================================

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

# In_jets=sys.argv[1] #Input file for jets
# In_subjets=sys.argv[2] #Input file for subjets
dir_jets_subjets_sig=sys.argv[1] #Input dir with files for jets and subjets
dir_jets_subjets_bg=sys.argv[2] #Input dir with files for jets and subjets of the set that I will use to get the standard deviation
# myN_jets=1000000000000000000000000000000000000000
myN_jets=5000
if(len(sys.argv)==4):
   myN_jets=int(sys.argv[3])


name_sig=dir_jets_subjets_sig.split('_')[1]
name_bg=dir_jets_subjets_bg.split('_')[1]

os.system('mkdir -p jet_array_1')
os.system('mkdir -p plots')
Images_dir=local_dir+'plots/' #Output dir to save the image plots
image_array_dir=local_dir+'jet_array_1/' #Output dir to save the image arrays

# bias kurtosis
# bias=5e-04
#-----
bias=2e-02
# bias=0.0


# bias=2e-02 #Value added to the standard deviation of each pixel over the whole training+test set before dividing the pixel value by the (standard deviation+bias) Comment: I was using 1e-03, but when looking at 1 jet images, this noise suppression value was so small that dividing by the standard deviation would totally change the location of the pixels with maximum intensity. So the best balance I found so far that puts pixels on a more equal level while keeping the location of the pixels with greatest intensity is 2e-02 

# npoints = 6 #npoint=(Number of pixels+1) of the image
npoints = 38 #npoint=(Number of pixels+1) of the image
DR=1.6 #Sets the size of the image as (2xDR,2xDR)
treshold=0.95 #We ask some treshold for the total pT fraction to keep the image when some constituents fall outside of the range for (eta,phi)
ptjmin=800 #Cut on the minimum pT of the jet
ptjmax=900 #Cut on the maximum pT of the jet
jetMass_min=130 #Cut on the minimum mass for the jet 
jetMass_max=210 #Cut on the maximum mass of the jet
# N_analysis=79 #Number of input files I want to include in the analysis
# N_analysis_sig=60 #Number of input files I want to include in the analysis
# N_analysis_bg=90 #Number of input files I want to include in the analysis

# N_analysis=8 #Number of input files I want to include in the analysis (For ~19000 tt images)
# N_analysis=5 #Number of input files I want to include in the analysis (For ~19000 QCD images)
#myN_jets=100000

sample_name='pflow'

signal='tt'
background='QCD'
N_pixels=np.power(npoints-1,2)


# std_label='own_std'
# std_label='avg_std'
# std_label='sig_std'
std_label='bg_std'
# std_label='no_std'
# std_label='stack_sig_bg_std'

# myMethod='std'
# myMethod='std'
myMethod='n_moment'


##=============================================================================================
############       FUNCTIONS TO LOAD, CREATE AND PREPROCESS THE JET IMAGES
##=============================================================================================

##---------------------------------------------------------------------------------------------
# 1) We load .npy files with jets and jet constituents (subjets) lists of [[pT],[eta],[phi]].
def loadfiles(jet_subjet_folder):
  print('Loading files for jet and subjets')
  print('Jet array format([[pTj1,pTj2,...],[etaj1,etaj2,...],[phij1,phij2,...],[massj1,massj2,...]])')
  print('Subjet array format ([[[pTsubj1],[pTsubj2],...],[[etasubj1],[etasubj2],...],[[phisubj1],[phisubj2],...]])')
  print('-----------'*10)
  
#  jetlist = [filename for filename in np.sort(os.listdir(jet_subjet_folder)) if filename.startswith('jets_')]
#  print('Jet files loaded = \n {}'.format(jetlist[0:N_analysis]))
#  subjetlist = [filename for filename in np.sort(os.listdir(jet_subjet_folder)) if filename.startswith('subjets_')]
#  print('Subjet files loaded = \n {}'.format(subjetlist[0:N_analysis]))

  jetlist = [filename for filename in np.sort(os.listdir(jet_subjet_folder)) if ('jets' in filename and filename.endswith('.npy') and 'subjets' not in filename)]
  N_analysis=len(jetlist)
  print('N_analysis =',N_analysis)
  print('Jet files loaded = \n {}'.format(jetlist[0:N_analysis]))
  subjetlist = [filename for filename in np.sort(os.listdir(jet_subjet_folder)) if ('subjets' in filename and filename.endswith('.npy'))]
  print('Subjet files loaded = \n {}'.format(subjetlist[0:N_analysis]))
  
  print('-----------'*10)
  print('Number of files loaded={}'.format(N_analysis))
#  print('Total number of files that could be loaded={}'.format(len(jetlist)))
  print('-----------'*10)
# 
#   print('len(jetlist)={}'.format(len(jetlist)))
#   print('len(subjetlist)={}'.format(len(subjetlist)))
  
  Jets=[] #List of jet files we are going to load
  for ijet in range(N_analysis):
#     Jets.append([])
    Jets.append(np.load(jet_subjet_folder+'/'+jetlist[ijet]))#We load the .npy files
  
  Alljets=[[],[],[],[]] # Format: [[pT],[eta],[phi],[mass]]
#   Alljets=[[],[],[]] # Format: [[pT],[eta],[phi]]
#   Each file has a tuple of ([[pTj1,pTj2,...],[etaj1,etaj2,...],[phij1,phij2,...],[massj1,massj2,...]) where in each element we have the data of many jets
  for file in range(N_analysis):
#     Alljets.append([])
    for tuple_element in range(len(Jets[file])): #The tuple_element is each element in ([pT],[eta],[phi],[mass])
#       row.append([])
      for ijet in range(len(Jets[file][tuple_element])):
        if ptjmin<Jets[file][0][ijet]<ptjmax and jetMass_min<Jets[file][3][ijet]<jetMass_max:
          Alljets[tuple_element].append(Jets[file][tuple_element][ijet])
  Alljets=np.array(Alljets)

#   print('Jets=\n {}'.format(Jets))     
#   print('Alljets (new way)=\n {}'.format(Alljets))

  Subjets=[]  #List of subjet files we are going to load
  for isubjet in range(N_analysis):
#     Subjets.append([])
    Subjets.append(np.load(jet_subjet_folder+'/'+subjetlist[isubjet]))#We load the .npy files
#     print('Dimension on subjets={}'.format(Subjets[isubjet].size))
#     print('lenght subjet',len(Subjets[isubjet]))
#     
#     print('Total lenght subjet',len(Subjets[isubjet]))
#     print('lenght subjet[0]=\n',len(Subjets[isubjet][0]))
    
  Allsubjets=[[],[],[]]
  for file in range(N_analysis):
#     Alljets.append([])
    for tuple_element in range(len(Subjets[file])):
#       row.append([])
      for ijet in range(len(Subjets[file][tuple_element])):
#         Allsubjets[tuple_element].append([])
#         for isubjet in range(len(Subjets[file][tuple_element][ijet])):
        if ptjmin<Jets[file][0][ijet]<ptjmax and jetMass_min<Jets[file][3][ijet]<jetMass_max:
          Allsubjets[tuple_element].append(Subjets[file][tuple_element][ijet])  
  
  Allsubjets=np.array(Allsubjets)  
#   print('Allsubjets (new way)=\n {}'.format(Allsubjets))  
#   print('-----------'*10)
#   print('-----------'*10)

  Njets=Alljets[0].size
  print('Njets = {}'.format(Njets)) 
  print('Nsubjets = {}'.format(Allsubjets[0].size)) 
  print('-----------'*10)
  
  return Alljets, Allsubjets, Njets


##---------------------------------------------------------------------------------------------
#2) We find the minimum angular distance (in phi) between jet constituents
def deltaphi(phi1,phi2):
   deltaphilist=[phi1-phi2,phi1-phi2+np.pi*2.,phi1-phi2-np.pi*2.]
   sortind=np.argsort(np.abs(deltaphilist))
   return deltaphilist[sortind[0]]


##---------------------------------------------------------------------------------------------
#3) We want to center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0). So we calculate eta_center,phi_center
def center(Subjets):
  print('Calculating the image center for the total pT weighted centroid pixel is at (eta,phi)=(0,0) ...')
  print('-----------'*10)
  #print('subjet type {}'.format(type(subjets[0][0])))

  Njets=len(Subjets[0])
  pTj=[]
  for ijet in range(0,Njets):  
    pTj.append(np.sum(Subjets[0][ijet]))
  #print('Sum of pTj for subjets = {}'.format(pTj))
  #print('pTj ={}'.format(jets[0][0])) #This is different for Sum of pTj for subjets, as for the jets, we first sum the 4-momentum vectors of the subjets and then get the pT
  #print('subjet 1 size {}'.format(subjets[1][0]))

  eta_c=[]
  phi_c=[]
  weigh_eta=[]
  weigh_phi=[]
  for ijet in range(0,Njets):
    weigh_eta.append([ ])
    weigh_phi.append([ ])
    for isubjet in range(0,len(Subjets[0][ijet])):
      weigh_eta[ijet].append(Subjets[0][ijet][isubjet]*Subjets[1][ijet][isubjet]/pTj[ijet]) #We multiply pT by eta of each subjet
      # print('weighted eta ={}'.format(weigh_eta))  
      weigh_phi[ijet].append(Subjets[0][ijet][isubjet]*deltaphi(Subjets[2][ijet][isubjet],Subjets[2][ijet][0])/pTj[ijet]) #We multiply pT by phi of each subjet
    eta_c.append(np.sum(weigh_eta[ijet])) #Centroid value for eta
    phi_c.append(np.sum(weigh_phi[ijet])+Subjets[2][ijet][0]) #Centroid value for phi
  #print('weighted eta ={}'.format(weigh_eta))
  #print('Position of pT weighted centroid pixel in eta for [jet1,jet2,...] ={}'.format(eta_c))
  #print('Position of pT weighted centroid pixel in phi for [jet1,jet2,...]  ={}'.format(phi_c))
  #print('-----------'*10)
  return pTj, eta_c, phi_c


##---------------------------------------------------------------------------------------------
#4) We shift the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates
def shift(Subjets,Eta_c,Phi_c):
  print('Shifting the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates ...')
  print('-----------'*10)

  Njets=len(Subjets[1])
  for ijet in range(0,Njets):
    if ijet == 0:
       print("center",Eta_c[ijet],Phi_c[ijet])
    Subjets[1][ijet]=(Subjets[1][ijet]-Eta_c[ijet])
    Subjets[2][ijet]=(Subjets[2][ijet]-Phi_c[ijet])
    Subjets[2][ijet]=np.unwrap(Subjets[2][ijet])#We fix the angle phi to be between (-Pi,Pi]
  #print('Shifted eta = {}'.format(Subjets[1]))
  #print('Shifted phi = {}'.format(Subjets[2]))
  #print('-----------'*10)
  return Subjets
  
  
##---------------------------------------------------------------------------------------------
#5) We calculate the angle theta of the principal axis
def principal_axis(Subjets):
  print('Getting DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis ...')
  print('-----------'*10) 
  tan_theta=[]#List of the tan(theta) angle to rotate to the principal axis in each jet image
  Njets=len(Subjets[1])
  for ijet in range(0,Njets):
     M11=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[2][ijet])
     M20=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[1][ijet])
     M02=np.sum(Subjets[0][ijet]*Subjets[2][ijet]*Subjets[2][ijet])
     tan_theta_use=2*M11/(M20-M02+np.sqrt(4*M11*M11+(M20-M02)*(M20-M02)))
     tan_theta.append(tan_theta_use)

     if ijet == 0:
       print("principal axis",tan_theta)
#   print('tan(theta)= {}'.format(tan_theta))
#   print('-----------'*10)
  return tan_theta


##---------------------------------------------------------------------------------------------
#6) We rotate the coordinate system so that the principal axis is the same direction (+ eta) for all jets
def rotate(Subjets,tan_theta):
  print('Rotating the coordinate system so that the principal axis is the same direction (+ eta) for all jets ...')
  print('-----------'*10)
#   print(Subjets[2][0])
#   print('Shifted eta for jet 1= {}'.format(Subjets[1][0]))
#   print('Shifted phi for jet 1 = {}'.format(Subjets[2][0]))
#   print('-----------'*10)
  rot_subjet=[[],[],[]]
  Njets=len(Subjets[1])
  for ijet in range(0,Njets):
    rot_subjet[0].append(Subjets[0][ijet]) 
    rot_subjet[1].append(Subjets[1][ijet]*np.cos(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.sin(np.arctan(tan_theta[ijet])))
    rot_subjet[2].append(-Subjets[1][ijet]*np.sin(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.cos(np.arctan(tan_theta[ijet])))
    #print('Rotated phi for jet 1 before fixing -pi<theta<pi = {}'.format(Subjets[2][0]))    
    rot_subjet[2][ijet]=np.unwrap(rot_subjet[2][ijet]) #We fix the angle phi to be between (-Pi,Pi]
#   print('Subjets pT (before rotation) = {}'.format(Subjets[0]))
#   print('-----------'*10)
#   print('Subjets pT (after rotation) = {}'.format(rot_subjet[0]))
#   print('-----------'*10)
#   print('eta = {}'.format(Subjets[1]))
#   print('-----------'*10)
#   print('Rotated eta = {}'.format(rot_subjet[1]))
#   print('-----------'*10)
#   print('Rotated phi = {}'.format(Subjets[2]))
#   print('-----------'*10)
#   print('Rotated phi = {}'.format(rot_subjet[2]))
#   print('-----------'*10)
#   print('-----------'*10)
  return rot_subjet
  

##---------------------------------------------------------------------------------------------
#7) We scale the pixel intensities such that sum_{i,j} I_{i,j}=1
def normalize(Subjets,pTj):
  print('Scaling the pixel intensities such that sum_{i,j} I_{i,j}=1 ...')
  print('-----------'*10)
  Njets=len(Subjets[0])
#   print('pT jet 2= {}'.format(Subjets[0][1])) 
  for ijet in range(0,Njets):
    Subjets[0][ijet]=Subjets[0][ijet]/pTj[ijet]
#   print('Normalizes pT jet 2= {}'.format(Subjets[0][1]))  
#   print('Sum of normalized pT for jet 2 = {}'.format(np.sum(Subjets[0][1])))
  print('-----------'*10)
  return Subjets



##---------------------------------------------------------------------------------------------
#8) We create a coarse grid for the array of pT for the jet constituents, where each entry represents a pixel. We add all the jet constituents that fall within the same pixel 
def create_image(Subjets):
    
  print('Generating images of the jet pT ...')
  print('-----------'*10)
  etamin, etamax = -DR, DR # Eta range for the image
  phimin, phimax = -DR, DR # Phi range for the image
  eta_i = np.linspace(etamin, etamax, npoints) #create an array with npoints elements between min and max
  phi_i = np.linspace(phimin, phimax, npoints)
  image=[]
  Njets=len(Subjets[0])
  print(Njets)
  for ijet in range(0,Njets):
    
    
    grid=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image 
#     print('Grid= {}'.format(grid))
#     print('eta_i= {}'.format(eta_i))
    
    eta_idx = np.searchsorted(eta_i,Subjets[1][ijet]) # np.searchsorted finds the index where each value in my data (Subjets[1] for the eta values) would fit into the sorted array eta_i (the x value of the grid).
    phi_idx = np.searchsorted(phi_i,Subjets[2][ijet])# np.searchsorted finds the index where each value in my data (Subjets[2] for the phi values) would fit into the sorted array phi_i (the y value of the grid).
  
#     print('Index eta_idx for jet {} where each eta value of the jet constituents in the data fits into the sorted array eta_i = \n {}'.format(ijet,eta_idx))
#     print('Index phi_idx for jet {} where each phi value of the jet constituents in the data fits into the sorted array phi_i = \n {}'.format(ijet,phi_idx))
#     print('-----------'*10)
    
#     print('Grid for jet {} before adding the jet constituents pT \n {}'.format(ijet,grid))
    for pos in range(0,len(eta_idx)):
      if eta_idx[pos]!=0 and phi_idx[pos]!=0 and eta_idx[pos]<npoints and phi_idx[pos]<npoints: #If any of these conditions are not true, then that jet constituent is not included in the image. 
        grid[eta_idx[pos]-1,phi_idx[pos]-1]=grid[eta_idx[pos]-1,phi_idx[pos]-1]+Subjets[0][ijet][pos] #We add each subjet pT value to the right entry in the grid to create the image. As the values of (eta,phi) should be within the interval (eta_i,phi_i) of the image, the minimum eta_idx,phi_idx=(1,1) to be within the image. However, this value should be added to the pixel (0,0) in the grid. That's why we subtract 1.         
#     print('Grid for jet {} after adding the jet constituents pT \n {}'.format(ijet,grid))  
#     print('-----------'*10)
    
    sum=np.sum(grid)
#     print('Sum of all elements of the grid for jet {} = {} '.format(ijet,sum))
#     print('-----------'*10)
#     print('-----------'*10)
    
    #We ask some treshold for the total pT fraction to keep the image when some constituents fall outside of the range for (eta,phi)
    
    if sum>=treshold:
# and ptjmin<Jets[0][ijet]<ptjmax and jetMass_min<Jets[3][ijet]<jetMass_max:
#       print('Jet Mass={}'.format(Jets[3][ijet]))
      image.append(grid)
      if len(grid)==0:
          print(ijet)
    if ijet%10000==0:
      print('Already generated jet images for {} jets'.format(ijet))
#   print('Array of images before deleting empty lists = \n {}'.format(image)) 
#   print('-----------'*10)
#  image=[array for array in image if array!=[]] #We delete the empty arrays that come from images that don't satisfy the treshold
  
#   print('Array of images = \n {}'.format(image[0:2])) 
#   print('-----------'*10)
  print('Number of images= {}'.format(len(image)))
  print('-----------'*10)
  N_images=len(image)
  
  Number_jets=N_images   #np.min([N_images, myN_jets])
  final_image=image
#  final_image=image[0:Number_jets]
  print('N_images = ',N_images)
  print('Final images = ',len(final_image))
  
  return final_image, Number_jets



##---------------------------------------------------------------------------------------------
#9) We subtract the mean mu_{i,j} of each image, transforming each pixel intensity as I_{i,j}=I_{i,j}-mu_{i,j}
def zero_center(Image,ref_image):
  print('Subtracting the mean mu_{i,j} of each image, transforming each pixel intensity as I_{i,j}=I_{i,j}-mu_{i,j} ...')
  print('-----------'*10)
  mu=[]
  Im_sum=[]
  N_pixels=np.power(npoints-1,2)
#  for ijet in range(0,len(Image)):
#    mu.append(np.sum(Image[ijet])/N_pixels)
#     Im_sum.append(np.sum(Image[ijet]))
#   print('Mean values of images= {}'.format(mu))
#   print('Sum of image pT (This should ideally be 1 as the images are normalized except when some jet constituents fall outside of the image range )= {}'.format(Im_sum))
  zeroImage=[]
  for ijet in range(0,len(Image)):# As some jet images were discarded because the total momentum of the constituents within the range of the image was below the treshold, we use len(image) instead of Njets
#     if ijet==10:
#         for i in range(37):
#            for j in range(37):
#               print("zero_center image")
#               print(i,j,Image[ijet][i,j])
#               print("ref image")
#               print(i,j,ref_image[i,j])
#               print("diff")
#               print((Image[ijet]-ref_image)[i,j])


    zeroImage.append(Image[ijet]-ref_image)
#    print(ijet,mu[ijet])

  print('Grid after subtracting the mean (1st 2 images)= \n {}'.format(Image[0:2])) 
  print('-----------'*10)
#   print('Mean of first images',mu[0:6])
  return zeroImage 


##---------------------------------------------------------------------------------------------
#10)Reflect the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane
def flip(Image,Nimages): 
  
  count=0
  print('Flipping the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane ...')
  print('-----------'*10)
  print('Image shape = ', np.shape(Image[0]))
#   print('Number of rows = ',np.shape(Image[0][0])[0])
  half_img=np.int((npoints-1)/2)
  flip_image=[]
  for i_image in range(len(Image)):
    left_img=[] 
    right_img=[]
    for i_row in range(np.shape(Image[i_image])[0]):
      left_img.append(Image[i_image][i_row][0:half_img])
      right_img.append(Image[i_image][i_row][-half_img:])
#       print('-half_img = ',-half_img)
#     print('Left half of image (we suppose the number of pixels is odd and we do not include the central pixel)\n',np.array(left_img))
#     print('Right half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(right_img))
     
    left_sum=np.sum(left_img)
    right_sum=np.sum(right_img)
#     print('Left sum = ',left_sum)
#     print('Right sum = ',right_sum)
    
    if left_sum>right_sum:
      flip_image.append(np.fliplr(Image[i_image]))     
    else:
      flip_image.append(Image[i_image])
#       print('Image not flipped')
#       print('Left sum = ',left_sum)
#       print('Right sum = ',right_sum)
      count+=1
#     print('Array of images before flipping =\n {}'.format(Image[i_image])) 
#     print('Array of images after flipping =\n {}'.format(flip_image[i_image])) 
  print('Fraction of images flipped = ',(Nimages-count)/Nimages)
  print('-----------'*10)
  print('-----------'*10)
  return flip_image  
 
 
##---------------------------------------------------------------------------------------------
#11)Reflect the image with respect to the horizontal axis to ensure the 3rd maximum is on the top half-plane
def hor_flip(Image,Nimages): 
  
  count=0
  print('Flipping the image with respect to the horizontal axis to ensure the 3rd maximum is on the top half-plane ...')
  print('-----------'*10)
  print('Image shape = ', np.shape(Image[0]))
  print('Number of columns = ',np.shape(Image[0])[1])
  half_img=np.int((npoints-1)/2)
  hor_flip_image=[]
  for i_image in range(len(Image)):
    top_img=[] 
    bottom_img=[]
#     print('image',Image[i_image])
#     print('image',Image[i_image][0])
    for i_row in range(half_img):
#       for i_col in range(np.shape(Image[i_image][0])[1]):
        top_img.append(Image[i_image][i_row])
        bottom_img.append(Image[i_image][-i_row-1])
#         print('-i_row-1 = ',-i_row-1)
#     print('Top half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(top_img))
#     print('Bottom half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(bottom_img))
    top_sum=np.sum(top_img)
    bottom_sum=np.sum(bottom_img)
#     print('Top sum = ',top_sum)
#     print('Bottom sum = ',bottom_sum)
#     
    if bottom_sum>top_sum:
      hor_flip_image.append(np.flip(Image[i_image],axis=0))     
    else:
      hor_flip_image.append(Image[i_image])
#       print('Image not flipped')

      count+=1
#     print('Array of images before flipping =\n {}'.format(Image[i_image])) 
#     print('Array of images after flipping =\n {}'.format(hor_flip_image[i_image])) 
  print('Fraction of images horizontally flipped = ',(Nimages-count)/Nimages)
  print('-----------'*10)
  print('-----------'*10)
  return hor_flip_image  
 
##---------------------------------------------------------------------------------------------
#12) We output a tuple with the numpy arrays and true value of the images that we will use as input for our neural network
def output_image_array_data_true_value(Image,type,name):
  Nimages=len(Image)
  true_value=[]
  for iImage in range(0,len(Image)):
    if name==signal:
      true_value.append(np.array([1]))
    elif name==background:
      true_value.append(np.array([0]))
    else:
      print('The sample is neither signal nor background. Update the signal/bacground names accordingly.')
  
#   print('True value where (1,0) means signal and (0,1) background =\n{}'.format(true_value))
  
  output=list(zip(Image,true_value))
#   print('Input array for neural network, with format (Input array,true value)= \n {}'.format(output))
  
  print("Saving data in .npy format ...")
  array_name=str(name)+'_'+str(Nimages)+'_'+str(npoints-1)+'_'+type+'_'+sample_name
  
#   f = gzip.open(image_array_dir+array_name+'.pkl.gz', 'w')#pkl.gz format
#   pickle.dump(output, f)
#   f.close()
  
#   .npy format
  np.save(image_array_dir+array_name+'_.npy',Image)
  print('List of jet image arrays filename = {}'.format(image_array_dir+array_name+'_.npy'))
  print('-----------'*10)
#   print('Array {}={}'.format(array_name,Image))


  
##---------------------------------------------------------------------------------------------
#13) We plot all the images
def plot_all_images(Image, type):
  
#   for ijet in range(0,len(Image)):
  for ijet in range(1200,1230):
    imgplot = plt.imshow(Image[ijet], 'gnuplot', extent=[-DR, DR,-DR, DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
    plt.xlabel('$\eta^{\prime\prime}$')
    plt.ylabel('$\phi^{\prime\prime}$')
  #plt.show()
    fig = plt.gcf()
    plt.savefig(Images_dir+'1jet_images/Im_'+str(name)+'_'+str(npoints-1)+'_'+str(ijet)+'_'+type+'.png')
#   print(len(Image))
#   print(type(Image[0]))


##---------------------------------------------------------------------------------------------
#14) We add the images to get the average jet image for all the events
def add_images(Image):
  print('Adding the images to get the average jet image for all the events ...')
  print('-----------'*10)
  N_images=len(Image)
#   print('Number of images= {}'.format(N_images))
#   print('-----------'*10)
  avg_im=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image
  for ijet in range(0,len(Image)):
    avg_im=avg_im+Image[ijet]
    #avg_im2=np.sum(Image[ijet])
#   print('Average image = \n {}'.format(avg_im))
  print('-----------'*10)
#  print('Average image 2 = \n {}'.format(avg_im2))
  #We normalize the image
  Total_int=np.absolute(np.sum(avg_im))
  print('Total intensity of average image = \n {}'.format(Total_int))
  print('-----------'*10)
#  norm_im=avg_im/Total_int
  norm_im=avg_im/N_images
#   print('Normalized average image (by number of images) = \n {}'.format(norm_im))
#   print('Normalized average image = \n {}'.format(norm_im))
  print('-----------'*10)
  norm_int=np.sum(norm_im)
  print('Total intensity of average image after normalizing (should be 1) = \n {}'.format(norm_int))
  return norm_im

  
  
##---------------------------------------------------------------------------------------------
#15) We plot the averaged image
def plot_avg_image(Image, type,name,Nimages):
  print('Plotting the averaged image ...')
  print('-----------'*10)
#   imgplot = plt.imshow(Image[0], 'viridis')# , origin='upper', interpolation='none', vmin=0, vmax=0.5)  
  imgplot = plt.imshow(Image, 'gnuplot', extent=[-DR, DR,-DR, DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
  plt.xlabel('$\eta^{\prime\prime}$')
  plt.ylabel('$\phi^{\prime\prime}$')
  fig = plt.gcf()
  image_name=str(name)+'_avg_im_'+str(Nimages)+'_'+str(npoints-1)+'_'+type+'_'+sample_name+'.png'
  plt.savefig(Images_dir+image_name)
  print('Average image filename = {}'.format(Images_dir+image_name))
#   print(len(Image))
#   print(type(Image[0]))

##=============================================================================================
############       MAIN FUNCTIONS
##=============================================================================================

##---------------------------------------------------------------------------------------------
# A) Plots images
def plot_my_image(images,std_name,type):

  Nimages=len(images)

  average_im =add_images(images)
  plot_avg_image(average_im,type,std_name,Nimages)  
  # plot_avg_image(average_im,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax)+'_'+myMethod,std_name,Nimages)


##---------------------------------------------------------------------------------------------
# B) PREPROCESS IMAGES (center, shift, principal_axis, rotate, normalize, vertical flip, horizontal flip)
def preprocess(subjets,std_name):
  

  pTj, eta_c, phi_c=center(subjets)  
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  shift_subjets=shift(subjets,eta_c,phi_c)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  #print(shift_subjets)
  tan_theta=principal_axis(shift_subjets) 
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  rot_subjets=rotate(shift_subjets,tan_theta)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  norm_subjets=normalize(rot_subjets,pTj)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
 
  print('Generating raw images.. .')
  raw_image, Nimages=create_image(norm_subjets)  
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  ver_flipped_img=flip(raw_image,Nimages)  
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  hor_flipped_img=hor_flip(ver_flipped_img,Nimages)  
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
  
#   plot_my_image(raw_image,std_name,'_rot'+'_'+str(ptjmin)+'_'+str(ptjmax))

  # plot_my_image(hor_flipped_img,std_name,'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax))
  
#   hor_flipped_img=raw_image
  return hor_flipped_img
 
 
 
##---------------------------------------------------------------------------------------------
# C) GET STANDARD DEVIATION OF A SET OF IMAGES
def get_std(Image,method): 

  print('-----------'*10)
  print('-----------'*10)
  print('Calculating standard deviation with a noise suppression factor ...')
  print('-----------'*10)
  Image_row=[]
#   N_pixels=np.power(npoints-1,2)
  print('Number of pixels of the image =',N_pixels)
  print('-----------'*10)
#   Image[0].reshape((N_pixels))
  for i_image in range(len(Image)):
#     Image_row.append([])
#     print('i_image ={}'.format(i_image))
    Image_row.append(Image[i_image].reshape((N_pixels)))
#   print('Image arrays as rows (1st 2 images)=\n {}'.format(Image_row[0:2]))
  print('-----------'*10)
  Image_row=np.array(Image_row,dtype=np.float64)
  Image_row.reshape((len(Image),N_pixels))
#   print('All image arrays as rows of samples and columns of features (pixels) (for the 1st 2 images) =\n {}'.format(Image_row[0:2]))
  print('-----------'*10)
  print('-----------'*10)
#   standard_img=preprocessing.scale(Image_row)

  if method=='n_moment':
#     kurtosis=scipy.stats.kurtosis(Image_row,axis=0, fisher=False)
    n_moment=scipy.stats.moment(Image_row, moment=4, axis=0)
    standard_dev=np.std(Image_row,axis=0,ddof=1, dtype=np.float64)
    print('N moment  =\n {}'.format(n_moment[0:40]))
    print('-----------'*10)
    final_bias=np.power(n_moment,1/4)+bias
    print('////////'*10)
    print('Max final bias = \n',np.sort(final_bias, axis=None)[::-1][0:20])
    print('-----------'*10)
#     final_bias=n_moment/np.power(standard_dev,2)+bias
#     print('N moment/std with bias for =\n {}'.format(final_bias[0:40]))
#    standard_img=Image_row/final_bias
#    print('-----------'*10)
#    print('N moment images with bias (1st 2 image arrays as rows)=\n {}'.format(standard_img[0:2]))
  
  elif method=='std':  
    standard_dev=np.std(Image_row,axis=0,ddof=1, dtype=np.float64)
    print('Standard deviation  =\n {}'.format(standard_dev))
    print('-----------'*10)
    final_bias=standard_dev+bias
    
  final_bias=final_bias.reshape((npoints-1,npoints-1))
#   print('Standard deviation with bias for =\n {}'.format(final_bias))
  print('-----------'*10)
  
  return final_bias


##---------------------------------------------------------------------------------------------
# D) USE STANDARD DEVIATION FROM ANOTHER SET OF IMAGES
def standardize_bias_std_other_set(Image, input_std_bias): 
  print('-----------'*10)
  print('-----------'*10)
  print('Standardizing image with std from another set and a noise suppression factor ...')
  print('-----------'*10) 
  std_im_list=[]
  for i_image in range(len(Image)):
    std_im_list.append(Image[i_image]/input_std_bias)
    std_im_list[i_image]=std_im_list[i_image].reshape((npoints-1,npoints-1))
  print('-----------'*10)
  print('-----------'*10)
  return std_im_list


##---------------------------------------------------------------------------------------------
# E) PUT ALL TOGETHER  
def standardize_images(images,reference_images,method):

# CALCULATE STD DEVIATION OF REFERENCE SET
  print('CALCULATING STANDARD DEVIATIONS OF REFERENCE SET')
  out_std_bias=get_std(reference_images, method)
  print("std for pixel",out_std_bias[15,15])
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

# CALCULATE AVERAGE IMAGE OF REFERENCE SET
  print('CALCULATING AVERAGE IMAGE OF REFERENCE SET')
  out_avg_image=add_images(reference_images)
  print("avg for pixel",out_avg_image[15,15])
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

# ZERO CENTER
  print('ZERO CENTERING IMAGES')
#  image_zero=zero_center(images,out_avg_image)
  image_zero=images
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

# DIVIDE BY STANDARD DEVIATION
  print('STANDARDIZING IMAGES')
  standard_image=standardize_bias_std_other_set(image_zero,out_std_bias)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)

  return standard_image


##---------------------------------------------------------------------------------------------
# F) Output averaged image and final npy array
def output(images,std_name):

  Nimages=len(images)

  average_im =add_images(images)  
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)  
#   output_image_array_data_true_value(images,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax)+'_'+myMethod,std_name)   
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
#   plot_all_images(standard_image,'std_'+str(bias)+'_flip_')
#   plot_all_images(flipped_img,'flip')
#   plot_all_images(flipped_img,'no_std')
  plot_avg_image(average_im,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax)+'_'+myMethod,std_name,Nimages)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
  

##=============================================================================================
############       RUN FUNCTIONS
##=============================================================================================
 

if __name__=='__main__':

##---------------------------------------------------------------------------------------------
# LOAD FILES
  print('LOADING FILES...')
  jets_sig,subjets_sig, Njets_sig=loadfiles(dir_jets_subjets_sig) 
  jets_bg,subjets_bg, Njets_bg=loadfiles(dir_jets_subjets_bg)
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
##---------------------------------------------------------------------------------------------
# PREPROCESS IMAGES
  print('PREPROCESSING IMAGES...')
  images_sig=preprocess(subjets_sig,'tt')
  images_bg=preprocess(subjets_bg,'QCD')  
  myN_jets=np.min([len(images_sig),len(images_bg),myN_jets]) 
  print("Number of images (sig=bg) used for analysis",myN_jets)
  images_sig=images_sig[0:myN_jets]
  images_bg=images_bg[0:myN_jets]
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
##---------------------------------------------------------------------------------------------  
#   plot_my_image(images_sig,'tt','_no_rot'+'_'+str(ptjmin)+'_'+str(ptjmax))
#   plot_my_image(images_bg,'QCD','_no_rot'+'_'+str(ptjmin)+'_'+str(ptjmax))

##---------------------------------------------------------------------------------------------
# ZERO CENTER AND NORMALIZE BY STANDARD DEVIATION
  print('ZERO CENTERING AND NORMALIZING IMAGES BY STANDARD DEVIATIONS...')
  if std_label == 'avg_std':
    sig_image_norm = standardize_images(images_sig,images_sig+images_bg,myMethod)
    bg_image_norm = standardize_images(images_bg,images_sig+images_bg,myMethod)
  elif std_label == 'bg_std':
    sig_image_norm = standardize_images(images_sig,images_bg,myMethod)
    bg_image_norm = standardize_images(images_bg,images_bg,myMethod)
  elif std_label == 'sig_std':
    sig_image_norm = standardize_images(images_sig,images_sig,myMethod)
    bg_image_norm = standardize_images(images_bg,images_sig,myMethod)
  elif std_label == 'no_std':
    sig_image_norm=images_sig
    bg_image_norm=images_bg
  elapsed=time.time()-start_time
  print('elapsed time',elapsed)
##---------------------------------------------------------------------------------------------
# OUTPUT
  print('OUTPUT...')
  output(sig_image_norm,'tt')
  output(bg_image_norm,'QCD')
#  output(images_sig,'tt')
#  output(images_bg,'QCD')


##---------------------------------------------------------------------------------------------
  print('FINISHED.')

  print('-----------'*10)
  print("Code execution time = %s minutes" % ((time.time() - start_time)/60))
  print('-----------'*10) 
  

  
  
