# Deep-learning_jet-images
Deep learning and computer vision techniques to identify jet substructure from proton-proton collisions at the Large Hadron Collider:

In searches for new physics, with the current 13 TeV center of mass energy of LHC Run II, it has become of great importance to classify jets from signatures with boosted W, Z and Higgs bosons, as well as top quarks with respect to the main QCD background. In this paper we use computer vision with deep learning to build a classifier for boosted top jets at LHC, that could also be straightforwardly extended to other types of jets. In particular, we implement a convolutional neural network (CNN) to identify jet substructure in signal and background events. Our CNN inputs are jet images that consist of five channels, where each of them represents a color. The first three colors are given by the transverse momentum (pT) of neutral particles from the Hadronic Calorimeter (HCAL) towers, the pT of charged particles from the tracking system and the charged particles multiplicity. The last two colors specify the muon multiplicity and b quark tagging information. We show that our top tagger performs significantly better than previous top tagging classifiers. For instance, we achieve a 60% top tagging efficiency with a (FILL IN) mistag rate for jets with pT in the 800-900 GeV range. We also analyze the contribution to the classification accuracy of the colors and a set of image preprocessing steps. Finally, we study the behavior of our method over two pT ranges and different event generators, i.e. PYTHIA and Herwig.


Description:

1) Simulations-Event_generation: Code to generate simulations of proton-proton collisions at the Large Hadron Collider that will be used to create the input images for the convolutional neural network.

2) image_preprocess.py loads the simulated jet and jet constituents, creates and preprocesses the images for the convnet. 

3) convnet_keras.py loads the (image arrays,true_values) tuples, creates the train, cross-validation and test sets and runs a convolutional neural network to classify signal vs background images. We then get the statistics and analyze the output. We plot histograms with the probability of signal and background to be tagged as signal, ROC curves and get the output of the intermediate layers and weights.
