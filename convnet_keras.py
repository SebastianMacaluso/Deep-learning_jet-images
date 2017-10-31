##=============================================================================================
##=============================================================================================
# IMPLEMENTATION OF A CONVOLUTIONAL NEURAL NETWORK TO CLASSIFY JET IMAGES AT LHC
##=============================================================================================
##=============================================================================================

# This script loads the (image arrays,true_values) tuples, creates the train, cross-validation and test sets and runs a convolutional neural network to classify signal vs background images. We then get the statistics and analyze the output. We plot histograms with the probability of signal and background to be tagged as signal, ROC curves and get the output of the intermediate layers and weights.
# Last updated: October 30, 2017. Sebastian Macaluso

##---------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------
# This code is ready to use on the jet_array1/test_large_sample dir. (The "expand image" function is currently commented). This version is for gray scale images.

# To run:
# Previous runs:
# python cnn_keras_jets.py input_sample_signal input_sample_background number_of_epochs fraction_to_use mode(train or notrain) weights_filename 

# python convnet_keras.py test_large_sample 20 0.1 train &> test2

##=============================================================================================
##=============================================================================================

##=============================================================================================
############       LOAD LIBRARIES
##=============================================================================================


from __future__ import print_function

import numpy as np
np.random.seed(1560)  # for reproducibility
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K # We are using TensorFlow as Keras backend
# from keras import optimizers
from keras.utils import np_utils

import pickle
import gzip
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import h5py

import time
start_time = time.time()

import data_loader as dl

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

##=============================================================================================
############       GLOBAL VARIABLES
##=============================================================================================

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

os.system('mkdir -p jet_array_3')

image_array_dir_in=local_dir+'jet_array_1/' #Input dir to load the array of images
# image_array_dir_in='../David/jet_array_1/'

in_arrays_dir= sys.argv[1]


# large_set_dir=image_array_dir_in+in_arrays_dir+'/'
large_set_dir=image_array_dir_in+in_arrays_dir+'/'

in_std_label='no_std' #std label of input arrays
std_label='bg_std' #std label for the standardization of the images with probabilities between prob_min and prob_max
bias=2e-02
npoints = 38 #npoint=(Number of pixels+1) of the image
N_pixels=np.power(npoints-1,2)
myMethod='std'
my_bins=20
# npoints=38

# extra_label='batch_norm'
extra_label='_early_stop'
min_prob=0.2
max_prob=0.8
my_batch_size = 128
num_classes = 2
epochs =int(sys.argv[2])
#Run over different sets sizes
sample_relative_size=float(sys.argv[3])

mode=sys.argv[4]
# mode='train'
# mode='notrain'

# input image dimensions
img_rows, img_cols = 37, 37

learning_rate=[np.sqrt(8.0)]# The default value for the learning rate (lr) Adadelta is 1.0. We divide the learning rate by sqrt(2) when the loss does not improve, we should start with lr=sqrt(8), so that the starting value is 2 (This is because we defined self.losses = [1,1] as the starting point). 
# learning_rate=[1.0]


##=============================================================================================
############       FUNCTIONS TO LOAD AND CREATE THE TRAINING, CROSS-VAL AND TEST SETS
##=============================================================================================

#1) We load the .npy file with the image arrays
def load_array(Array):
  print('Loading signal and background arrays ...')
  print('-----------'*10)
  data=np.load(large_set_dir+Array) #We load the .npy files
  return data

##---------------------------------------------------------------------------------------------
#2) We expand the images (adding zeros when necessary)
def expand_array(images):
# ARRAY MUST BE IN THE FORM [[[iimage,ipixel,jpixel],val],...]

  Nimages=len(images)

  print('Number of images ',Nimages)

  expandedimages=np.zeros((Nimages,img_rows,img_cols))

  for i in range(Nimages):
#    print(i,len(images[i]))
    for j in range(len(images[i])):
#       print(i,j,images[i][j][1])
       expandedimages[images[i][j][0][0],images[i][j][0][1],images[i][j][0][2]] = images[i][j][1]
#  np.put(startgrid,ind,val)

  return expandedimages

##---------------------------------------------------------------------------------------------
#3) We create a tuple of (image array, true value) joining signal and background, and we shuffle it.
def add_sig_bg_true_value(Signal,Background):
  print('Creating tuple (data,true value) ...')
  print('-----------'*10)
  Signal=np.asarray(Signal)
  Background=np.asarray(Background)
  input_array=[]
  true_value=[]
  for ijet in range(0,len(Signal)):
    input_array.append(Signal[ijet].astype('float32'))
    true_value.append(np.array([0]).astype('float32'))
#   print('List of arrays for signal = \n {}'.format(input_array))
#   print('-----------'*10)

  for ijet in range(0,len(Background)):
    input_array.append(Background[ijet].astype('float32'))
    true_value.append(np.array([1]).astype('float32'))
#   print('Joined list of arrays for signal and background = \n {}'.format(input_array))
#   print('-----------'*10)
#   print('Joined list of true values for signal and background = \n {}'.format(true_value))
#   print('-----------'*10)

  output=list(zip(input_array,true_value))
  #print('Input array for neural network, with format (Input array,true value)= \n {}'.format(output[0][0]))
#   for (x,y) in output:
#    print('x={}'.format(x))
#    print('y={}'.format(y))
#   print('-----------'*10)
  print('Shuffling tuple (data, true value) ...')
  print('-----------'*10)
  shuffle_output=np.random.permutation(output)

  return shuffle_output

##---------------------------------------------------------------------------------------------
#4) This function loads the zipped tuple of image arrays and true values. It divides the data into train and validation sets. Then we create new arrays with$
def sets(Data):
  print('Generating arrays with the correct input format for Keras ...')
  print('-----------'*10)
#   Ntrain=int(0.8*len(Data))
  X=[x for (x,y) in Data]
  Y=[y for (x,y) in Data]
#   Y_test=[y for (x,y) in Data[Ntrain:]]

#   print('X (train+test) before adding [] to each element=\n{}'.format(X))
  X=np.asarray(X)
  print('Shape X = {}'.format(X.shape))
  X_out=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
  print('-----------'*10)
  print('Shape X out after adding [] to each element= {}'.format(X_out.shape))
#  print('Input arrays X_out after adding [] to each element (middle row)=\n{}'.format(X_out[17][0:37]))
  print('-----------'*10)

  output_tuple=list(zip(X_out,Y))

#   print('Tuple of (array,true value) as input for the cnn =\n {}'.format(output_tuple))

  return output_tuple



##---------------------------------------------------------------------------------------------
#5) Get the list with the input images file names
def get_input_array_list(input_array_dir):
#   sg_imagelist = [filename for filename in np.sort(os.listdir(input_array_dir)) if filename.startswith('tt_')  and 'batch' in filename and 210000>int(filename.split('_')[1])>190000]
#   bg_imagelist = [filename for filename in np.sort(os.listdir(input_array_dir)) if filename.startswith('QCD_')  and 'batch' in filename and 210000>int(filename.split('_')[1])>190000]

  sg_imagelist = [filename for filename in np.sort(os.listdir(input_array_dir)) if filename.startswith('tt_') ] # and 'batch' in filename and 210000>int(filename.split('_')[1])>190000]
  bg_imagelist = [filename for filename in np.sort(os.listdir(input_array_dir)) if filename.startswith('QCD_')]  # and 'batch' in filename and 210000>int(filename.split('_')[1])>190000]

#     N_arrays=len(imagelist)
  return sg_imagelist, bg_imagelist

##---------------------------------------------------------------------------------------------
#6) Define a dictionary to identify the training, cross-val and test sets
def load_all_files(array_list):
  dict={}
  for index in range(len(array_list)):
    dict[index]=load_array(array_list[index])
    print('Dict {} lenght = {}'.format(index,len(dict[index])))
  
  return dict

##---------------------------------------------------------------------------------------------
#7) Cut the number of images in the sample when necessary
def cut_sample(data_tuple, sample_relative_size):

  print('-----------'*10)
  print(data_tuple.shape, 'Input array sample shape before cut') 
  print('-----------'*10)
  N_max= int(sample_relative_size*len(data_tuple))
  out_array= data_tuple[0:N_max]
  print(out_array.shape, 'Input array sample shape after cut')
  print('-----------'*10)
  return out_array


##---------------------------------------------------------------------------------------------
#8) Split the sample into train, cross-validation and test
def split_sample(data_tuple, train_frac_rel, val_frac_rel, test_frac_rel):

  val_frac_rel=train_frac_rel+val_frac_rel
  test_frac_rel =(val_frac_rel+test_frac_rel)
  
  train_frac=train_frac_rel
  val_frac=val_frac_rel
  test_frac=test_frac_rel

  N_train=int(train_frac*len(data_tuple))
  Nval=int(val_frac*len(data_tuple))
  Ntest=int(test_frac*len(data_tuple))

  x_train=[x for (x,y) in data_tuple[0:N_train]]
  Y_train=[y for (x,y) in data_tuple[0:N_train]]
  x_val=[x for (x,y) in data_tuple[N_train:Nval]]
  Y_val=[y for (x,y) in data_tuple[N_train:Nval]]
  x_test=[x for (x,y) in data_tuple[Nval:Ntest]]
  Y_test=[y for (x,y) in data_tuple[Nval:Ntest]]

  ##---------------------------------------------------------------------------------------------
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(Y_train, num_classes)
  y_val = keras.utils.to_categorical(Y_val, num_classes)
  y_test = keras.utils.to_categorical(Y_test, num_classes)

  ##---------------------------------------------------------------------------------------------
  # Define input data format as Numpy arrays
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_val = np.array(x_val)
  y_val = np.array(y_val)
  x_test = np.array(x_test)
  y_test = np.array(y_test)

  # x_train = x_train.astype('float32')
  # print('x_train = \n {}'.format(x_train))
  # print('x_train shape:', x_train[0].shape)

  print('-----------'*10)
  print(len(x_train), 'train samples ('+str(train_frac*100)+'% of the set)')
  print(len(x_val), 'validation samples ('+str((val_frac-train_frac)*100)+'% of the set)')
  print(len(x_test), 'test samples ('+str(100-(val_frac)*100)+'% of the set)')
  print('-----------'*10)

  print(x_train.shape, 'train sample shape') # train_x.shape should be (batch or number of samples, height, width, channels), where channels is 1 for gray scale and 3 for RGB pictures
  print('-----------'*10)
  print('-----------'*10)

  # print('y_train=\n {}'.format(y_train))
  # print('y_test=\n {}'.format(y_test))
  return x_train, y_train, x_val, y_val, x_test, y_test#, N_train

##---------------------------------------------------------------------------------------------
#9) Concatenate arrays into a single set (i.e. cross-val or test) when multiple files are loaded
def concatenate_arrays(array_list, label_sg_bg, label):

  if label_sg_bg=='sg' and label=='val':
    temp_array=my_dict_val_sg[0]
  elif label_sg_bg=='bg' and label=='val':
    temp_array=my_dict_val_bg[0] 
  elif label_sg_bg=='sg' and label=='test':
    temp_array=my_dict_test_sg[0]    
  elif label_sg_bg=='bg' and label=='test':
    temp_array=my_dict_test_bg[0] 
  else:
    print('Please specify the right labels')
        
#   temp_array=load_array(array_list[0])
  temp_array = cut_sample(temp_array, sample_relative_size)
#   temp_array = expand_array(temp_array)
  
  for index in range(len(array_list[1::])):
  
    new_index=index+1
    if label_sg_bg=='sg' and label=='val':
      single_array=my_dict_val_sg[new_index]
    elif label_sg_bg=='bg' and label=='val':
      single_array=my_dict_val_bg[new_index] 
    elif label_sg_bg=='sg' and label=='test':
      single_array=my_dict_test_sg[new_index]    
    elif label_sg_bg=='bg' and label=='test':
      single_array=my_dict_test_bg[new_index] 
    else:
      print('Please specify the right labels')
  
#     single_array= load_array(i_file)
    single_array = cut_sample(single_array, sample_relative_size)
#     single_array=expand_array(single_array)
    elapsed=time.time()-start_time
    print('images expanded')
    print('elapsed time',elapsed) 
    temp_array=np.concatenate((temp_array,single_array), axis=0)
  return temp_array


##---------------------------------------------------------------------------------------------
#10) Create the validation and test sets
def generate_input_sets(sg_files, bg_files,train_frac_rel, in_val_frac_rel,in_test_frac_rel, set_label):
  print('Generates batches of samples for {}'.format(sg_files))
  # Infinite loop
  print('len(sg_files)=',len(sg_files))
  indexes = np.arange(len(sg_files))

  print('-----------'*10)
  print( 'indexes =',indexes)  
  print('-----------'*10)
  
  
#   signal= load_array(sg_files[0])
#   background = load_array(bg_files[0])

  signal = concatenate_arrays(sg_files,'sg',set_label)
  background = concatenate_arrays(bg_files,'bg',set_label)

  print(signal.shape, 'signal sample shape') # train_x.shape should be (batch or number of samples, height, width, channels), where channels is 1 for gray scale and 3 for RGB pictures
  print(background.shape, 'background sample shape') # train_x.shape should be (batch or number of samples, height, width, channels), where channels is 1 for gray scale and 3 for RGB pictures

  data_in= add_sig_bg_true_value(signal,background)
  data_tuple = sets(data_in)

  x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = split_sample(data_tuple, train_frac_rel, in_val_frac_rel,in_test_frac_rel)

  if set_label=='train':
    print('-----------'*10)
    print('Using training dataset')
    print('-----------'*10)
    return x_train1, y_train1
  elif set_label == 'val':
    print('-----------'*10)
    print('Using validation dataset')
    print('-----------'*10)
    return x_val1, y_val1      
  elif set_label=='test':
    return x_test1, y_test1  

##=============================================================================================
############       DATA GENERATOR CLASS TO GENERATE THE BATCHES FOR TRAINING
##=============================================================================================
# (We create this class because the sample size is larger than the memory of the system)
# The data generator is just that a generator that has no idea how the data it generates is going to be used and at what epoch. It should just keep generating batches of data forever as needed.

class DataGenerator(object):
  print('Generates data for Keras')
  def __init__(self, dim_x = img_rows, dim_y = img_cols,  batch_size = my_batch_size, shuffle = False):
#       'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle


  def generate(self, sg_files, bg_files,train_frac_rel, in_val_frac_rel,in_test_frac_rel, set_label):
    print('Generates batches of samples for {}'.format(sg_files))
    # Infinite loop
#     print('len(sg_files)=',len(sg_files))
    
    while True:
      
      indexes = np.arange(len(sg_files))
      print('len(sg_files)= ',len(sg_files))
      for index in indexes:

        name_sg=str('_'.join(sg_files[index].split('_')[:2]))
        name_bg=str('_'.join(bg_files[index].split('_')[:-1]))
        in_tuple=name_sg+'_'+name_bg
#         print('Name signal ={}'.format(name_sg))
#         print('Name background={}'.format(name_bg))
#         print('-----------'*10)

        signal= my_dict_train_sg[index]
        background = my_dict_train_bg[index]

#         signal= load_array(sg_files[index])
#         background = load_array(bg_files[index])

        signal = cut_sample(signal, sample_relative_size)
        background = cut_sample(background, sample_relative_size)
        
#         signal=expand_array(signal)
#         background=expand_array(background)
        elapsed=time.time()-start_time
        print('images expanded')
        print('elapsed time',elapsed) 

        data_in= add_sig_bg_true_value(signal,background)
        data_tuple = sets(data_in)

        x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = split_sample(data_tuple, train_frac_rel, in_val_frac_rel,in_test_frac_rel)
        
        
        subindex= np.arange(len(x_train1))
        print('len(x_train1[{}])= {}'.format(index,len(x_train1)))
        imax = int(len(subindex)/self.batch_size)
        print('imax =',imax)
        print('\n'+'-----------'*10)
        print('////////////'*10)
        
        for i in range(imax):
      
          if set_label=='train':
#             x_train_temp = [x_train1[k] for k in subindex[i*self.batch_size:(i+1)*self.batch_size]]        
#             y_train_temp = [y_train1[k] for k in subindex[i*self.batch_size:(i+1)*self.batch_size]]
            x_train_temp = x_train1[i*self.batch_size:(i+1)*self.batch_size]        
            y_train_temp = y_train1[i*self.batch_size:(i+1)*self.batch_size] 

#             print(x_train_temp.shape, 'x_train_temp sample shape') # train_x.shape should be (batch or number of samples, height, width, channels), where channels is 1 for gray scale and 3 for RGB pictures
            
#             print('-----------'*10)
#             print('Using training dataset')
#             print('-----------'*10)
            yield x_train_temp, y_train_temp
            
          elif set_label == 'val':
#             x_val_temp = [x_val1[k] for k in subindex[i*self.batch_size:(i+1)*self.batch_size]]        
#             y_val_temp = [y_val1[k] for k in subindex[i*self.batch_size:(i+1)*self.batch_size]]
            x_val_temp = x_val1[i*self.batch_size:(i+1)*self.batch_size]       
            y_val_temp = y_val1[i*self.batch_size:(i+1)*self.batch_size]
            print('-----------'*10)
            print('Using validation dataset')
            print('-----------'*10)
            yield x_val_temp, y_val_temp   
               
          elif set_label=='test':
            yield x_test1, y_test1
    

##=============================================================================================
############       LOAD AND CREATE THE TRAINING, CROSS-VAL AND TEST SETS
##=============================================================================================
signal_array_list,background_array_list = get_input_array_list(large_set_dir)
#----------------------------------------------------------------------------------------------------

train_signal_array_list = signal_array_list[0:-4]
train_background_array_list = background_array_list[0:-4]
print('-----------'*10)
print('-----------'*10)
print('train_signal_array_list=',train_signal_array_list)
print('-----------'*10)
print('train_bg_array_list=',train_background_array_list)
print('-----------'*10)


total_images=0
for i_file in range(len(train_signal_array_list)):
  steps_file=load_array(train_signal_array_list[i_file])
  total_images+=len(steps_file)

# If the fraction from my input files for tratining is different from (train_frac, val_frac, test_frac)=(1,0,0), then also multiply Ntrain*train_frac
Ntrain=2*total_images*sample_relative_size
print('Ntrain',Ntrain)

val_signal_array_list = signal_array_list[-4:-2]
val_background_array_list = background_array_list[-4:-2]  

test_signal_array_list = signal_array_list[-2::]
test_background_array_list = background_array_list[-2::]

print('-----------'*10)
print('val_signal_array_list=',val_signal_array_list)
print('-----------'*10)
print('val_bg_array_list=',val_background_array_list)
print('-----------'*10)
print('-----------'*10)
print('test_signal_array_list=',test_signal_array_list)
print('-----------'*10)
print('test_bg_array_list=',test_background_array_list)
print('-----------'*10)

##---------------------------------------------------------------------------------------------
# Load all the files to the dictionary

my_dict_train_sg=load_all_files(train_signal_array_list)
my_dict_train_bg=load_all_files(train_background_array_list)

my_dict_val_sg=load_all_files(val_signal_array_list)
my_dict_val_bg=load_all_files(val_background_array_list)

my_dict_test_sg=load_all_files(test_signal_array_list)
my_dict_test_bg=load_all_files(test_background_array_list)


##=============================================================================================
############       DEFINE THE NEURAL NETWORK ARCHITECTURE AND IMPLEMETATION
##=============================================================================================

input_shape = (img_rows, img_cols,1)

model = Sequential()

convin1=Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=input_shape)
model.add(convin1)

convout1 = MaxPooling2D(pool_size=(2, 2))
model.add(convout1)

model.add(Conv2D(64, (4, 4), activation='relu'))

convout2=MaxPooling2D(pool_size=(2, 2))
model.add(convout2)

model.add(Dropout(0.25))
model.add(Conv2D(64, (2, 2), activation='relu'))

convout3=MaxPooling2D(pool_size=(2, 2))
model.add(convout3)

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

##---------------------------------------------------------------------------------------------
# LOSS FUNCTION - OPTIMIZER
# We define the loss/cost function and the optimizer to reach the minimum (e.g. gradient descent, adadelta, etc).
#a) For loss=keras.losses.categorical_crossentropy, we need to get the true values in the form of vectors of 0 and 1: y_train = keras.utils.to_categorical(y_train, num_classes) 
#b) Use metrics=['accuracy'] for classification problems

#1) Adadelta
Adadelta=keras.optimizers.Adadelta(lr=learning_rate[0], rho=0.95, epsilon=1e-08, decay=0.0)

#2) Adam
Adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#3) Sigmoid gradient descent: the convergence is much slower than with Adadelta
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adadelta,
              metrics=['categorical_accuracy'])


##---------------------------------------------------------------------------------------------
# FUNCTIONS TO ADJUST THE LEARNING RATE 
# We write functons to divide by 2 the learning rate when the validation loss (val_loss) does not improve within some treshold

# Get the validation losses after each epoch
sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1] #Initial value of the val loss function

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss')) # We append the val loss of the last epoch to losses
        sd.append(step_decay(len(self.losses))) # We run step_decay to determine if we update the learning rate
        # print('lr:', step_decay(len(self.losses)))

##-----------------------------
# Take the difference between the last 2 val_loss and divide the learning rate by sqrt(2) when it does not improve. Both requirements should be satisfied:
#1) loss[-2]-loss[-1]<0.0005
#2) loss[-2]-loss[-1]< loss[-1]/3
def step_decay(losses): 

#   if float(np.array(history.losses[-2])-np.array(history.losses[-1]))<0.0005 and
	if float(np.array(history.losses[-2])-np.array(history.losses[-1]))<0.0001 and float(np.array(history.losses[-2])-np.array(history.losses[-1]))< np.array(history.losses[-1])/3:
		print('\n loss[-2] = ',np.array(history.losses[-2]))
		print('\n loss[-1] = ',np.array(history.losses[-1]))
		print('\n loss[-2] - loss[-1] = ',float(np.array(history.losses[-2])-np.array(history.losses[-1])))
		lrate=learning_rate[-1]/np.sqrt(2)
		learning_rate.append(lrate)
	else:
		lrate=learning_rate[-1]

	print('\n Learning rate =',lrate)
	print('------------'*10)
	
	return lrate

##-----------------------------
history=LossHistory() #We define the class history that will have the val loss values
# Get val_loss for each epoch. This is called at the end of each epoch and it will append the new value of the val_loss to the list 'losses'. 
lrate=keras.callbacks.LearningRateScheduler(step_decay) # Get new learning rate

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=4, verbose=0, mode='auto')
# patience=4 means that if there is no improvement in the cross-validation accuracy greater that 0.002 within the following 3 epochs, then it stops


##=============================================================================================
############       TRAIN THE MODEL (OR LOAD TRAINED WEIGHTS)
##=============================================================================================
#Make folder to save weights
weights_dir = 'weights/'
#os.system("rm -rf "+executedir)
os.system("mkdir -p "+weights_dir)


if mode=='notrain':
  my_weights=sys.argv[6]
  WEIGHTS_FNAME=weights_dir+my_weights
  if True and os.path.exists(WEIGHTS_FNAME):
     # Just change the True to false to force re-training
      print('Loading existing weights')
      print('------------'*10)
      model.load_weights(WEIGHTS_FNAME)
  else: 
    print('Please specify a weights file to upload')
    
elif mode=='train':
  # We add history and lrate as callbacks. A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument callbacks) to the .fit() method of the Sequential or Model classes. The relevant methods of the callbacks will then be called at each stage of the training.
  #   my_weights_name='cnn_weights_epochs_'+str(epochs)+'_Ntrain_'+str(Ntrain)+'_'+in_tuple+extra_label+'.hdf'
  my_weights_name='cnn_weights_epochs_'+str(epochs)+'_Ntrain_'+str(Ntrain)+'_'+extra_label+'.hdf' 
  #Load weights and continue with training  
  if(len(sys.argv)>6):
    my_weights=sys.argv[6]
    WEIGHTS_FNAME=weights_dir+my_weights
    if True and os.path.exists(WEIGHTS_FNAME):
      # Just change the True to false to force re-training
        print('Loading existing weights')
        print('------------'*10)
        model.load_weights(WEIGHTS_FNAME)
    
    previous_epoch=int('_'.join(my_weights.split('_')[3]))    
    # my_weights_name='cnn_weights_epochs_'+str(epochs+previous_epoch)+'_Ntrain_'+str(Ntrain)+'_'+in_tuple+extra_label+'.hdf'  
    my_weights_name='cnn_weights_epochs_'+str(epochs+previous_epoch)+'_Ntrain_'+str(Ntrain)+'_'+extra_label+'.hdf'   

  ##-----------------------------
  # Create training and  cross-validation sets                  
  train_x_train_y = DataGenerator().generate(train_signal_array_list, train_background_array_list, 1.0,0.0,0.0, 'train')
  # val_x_val_y = DataGenerator().generate(val_signal_array_list, val_background_array_list, 0.0,1.0,0.0, 'val')
  
  val_x, val_y = generate_input_sets(val_signal_array_list, val_background_array_list, 0.0,1.0,0.0, 'val')
      
  print('total_images =',total_images)  
  my_steps_per_epoch= int(2*total_images*sample_relative_size/my_batch_size)  

  print('my_steps_per_epoch =',my_steps_per_epoch)

  my_max_q_size=my_steps_per_epoch/6

  ##-----------------------------
  # Run Keras training routine  
  model.fit_generator(generator = train_x_train_y,
                    steps_per_epoch = my_steps_per_epoch, #This is the number of files that we use to train in each epoch
                    epochs=epochs,
                    verbose=2,
                    validation_data =(val_x, val_y)
                    ,max_q_size=my_max_q_size  # defaults to 10
                    ,callbacks=[history,lrate,early_stop]
                     )                   

  WEIGHTS_FNAME = weights_dir+my_weights_name    
  print('------------'*10)
  print('Weights filename =',WEIGHTS_FNAME)
  print('------------'*10)
  # We save the trained weights
  model.save_weights(WEIGHTS_FNAME, overwrite=True)

else:
  print('Please specify a valid mode')
  
print('------------'*10)

##-----------------------------
# Create the test set and evaluate the model
test_x, test_y = generate_input_sets(test_signal_array_list, test_background_array_list, 0.0,0.0,1.0, 'test')
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss = ', score[0])
print('Test accuracy = ', score[1])

print('------------'*10)
print('All learning rates = ',learning_rate)
print('------------'*10)


# sys.exit()


##=============================================================================================
##=============================================================================================
###########################          ANALYZE RESULTS       ####################################
##=============================================================================================
##=============================================================================================


##=============================================================================================
############       LOAD LIBRARIES
##=============================================================================================
                                                                                                                    
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches

##=============================================================================================
############       GLOBAL VARIABLES
##=============================================================================================

N_out_layer0=1
N_out_layer1=32
N_out_layer2=64

##-------------------------------
name_sg=str('_'.join(signal_array_list[0].split('_')[:2]))
name_bg=str('_'.join(background_array_list[0].split('_')[:-1]))
in_tuple=name_sg+'_'+name_bg
print('------------'*10)
print('------------'*10)
print('in_tuple = ',in_tuple)
print('------------'*10)
name='_'.join(in_tuple.split('_')[:4])+'_pTj_'+'_'.join(in_tuple.split('_')[-3:-1])

# print(in_tuple.split('_'))
print('Name of dir with weights and output layer images=',name)

##-------------------------------
# Create directorires
os.system('mkdir -p analysis/')
os.system('mkdir -p analysis/outlayer_plots/')
os.system('mkdir -p analysis/weight_plots/')

##=============================================================================================
############       PREDICT OUTPUT PROBABILITIES
##=============================================================================================

# Predict output probability for each class (signal or background) for the image
Y_Pred_prob = model.predict(test_x)

print('y_Test (categorical). This is a vector of zeros with a one in the position of the image class =\n ',test_y[0:15])

# Convert vector of 1 and 0 to index
y_Pred = np.argmax(Y_Pred_prob, axis=1)
y_Test = np.argmax(test_y, axis=1)
print('Predicted output from the CNN (0 is signal and 1 is background) = \n',y_Pred[0:15])
print('y_Test (True value) =\n ',y_Test[0:15])
print('y_Test lenght', len(y_Test))
print('------------'*10)

#Print classification report
print(classification_report(y_Test, y_Pred))
print('------------'*10)

##---------------------------------------------------------------------------------------------  
# We calculate a single probability of tagging the image as signal
out_prob=[]
for i_prob in range(len(Y_Pred_prob)):
    out_prob.append((Y_Pred_prob[i_prob][0]-Y_Pred_prob[i_prob][1]+1)/2)

print('Predicted probability of each output neuron = \n',Y_Pred_prob[0:15])
print('------------'*10)
print('Output of tagging image as signal = \n',np.array(out_prob)[0:15])
print('------------'*10)


##----------------------------------------------------
#Make folder to save output probability and true values                                                                                                             
outprob_dir = 'analysis/out_prob/'
#os.system("rm -rf "+executedir)                                                                                                          
os.system("mkdir -p "+outprob_dir)


##  SAVE OUTPUT PROBABILITIES AND TRUE VALUES
# np.save(outprob_dir+'out_prob_'+in_tuple,out_prob)
# np.save(outprob_dir+'true_value_'+in_tuple,y_Test)

print('Output probabilitiy filename = {}'.format(outprob_dir+'out_prob_'+in_tuple))
print('True value filename = {}'.format(outprob_dir+'true_value_'+in_tuple))



##=============================================================================================
############       Analysis over the images in the "mistag range" 
# (images with prob between min_prob and max_prob)
##=============================================================================================

##------------------------------------------------------------------------------------------
# 1) Get probability for signal and background sets to be tagged as signal
# 2) Get index of signal and bg images with a prob of being signal in some specific range

# y_Test is the true value and out_prob the predicted probability of the image to be signal
sig_prob=[] #Values of the precicted probability that are labeled as signal in the true value array
bg_prob=[] #Values of the precicted probability that are labeled as bg in the true value array

sig_idx=[]
bg_idx=[]


for i_label in range(len(y_Test)):

  if y_Test[i_label]==0: #signal label
    sig_prob.append(out_prob[i_label])
    if min_prob<out_prob[i_label]<max_prob:
      sig_idx.append(i_label)
      
  elif y_Test[i_label]==1: #bg label
    bg_prob.append(out_prob[i_label])
    if min_prob<out_prob[i_label]<max_prob:
      bg_idx.append(i_label)
    
print('-----------'*10)
print('-----------'*10)
print('Predicted probability (images labeled as signal) = \n',sig_prob[0:15])
print('-----------'*10)
print('Predicted probability (images labeled as background) =\n ',bg_prob[0:15])
print('-----------'*10)
##--------------------------
# Get the array of bg and signal images with a prob of being signal within some specific range

sig_images=[]
bg_images=[]

sig_label=[]
bg_label=[]

for index in sig_idx:
  sig_images.append(test_x[index])
  sig_label.append(y_Test[index])

for index in bg_idx:
  bg_images.append(test_x[index])
  bg_label.append(y_Test[index])

sig_images=np.asarray(sig_images)
bg_images=np.asarray(bg_images)

print('-----------'*10)
print('Number of signal images in the slice between %s and %s = %i' %(str(min_prob), str(max_prob),len(sig_images)))
print('-----------'*10)
print('Number of background images in the slice between %s and %s = %i' %(str(min_prob), str(max_prob),len(bg_images)))
print('-----------'*10)
print('-----------'*10)
print('Signal images with a prob between %s and %s label (1st 10 values) = \n %a' % (str(min_prob), str(max_prob),sig_label[0:10]))
print('-----------'*10)
print('Background images with a prob between %s and %s label (1st 10 values) = \n %a' % (str(min_prob), str(max_prob),bg_label[0:10]))
print('-----------'*10)


##=============================================================================================
############       PLOT HISTOGRAM OF SIG AND BG EVENTS DEPENDING ON THEIR PROBABILITY OF BEING TAGGED AS SIGNAL
##=============================================================================================

#Make folder to save plots                                                                                                             
outprob_dir = 'analysis/out_prob/'
#os.system("rm -rf "+executedir)                                                                                                          
os.system("mkdir -p "+outprob_dir)

# Histogram function
def make_hist(in_sig_prob,in_bg_prob,name):
  # the histogram of the data
#   n, bins, patches = plt.hist(sig_prob, my_bins, facecolor='red')
#   n, bins, patches = plt.hist(bg_prob, my_bins, facecolor='blue')

  plt.hist(in_sig_prob, my_bins, alpha=0.5, facecolor='red')
  plt.hist(in_bg_prob, my_bins, alpha=0.5, facecolor='blue')


  red_patch = mpatches.Patch(color='red', label='True value = top jet')
  blue_patch = mpatches.Patch(color='blue', label='True value = qcd jet')
  plt.legend(handles=[red_patch,blue_patch],bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
  # plt.legend(handles=[red_patch,blue_patch])
  # plt.legend(handles=[blue_patch])
  # add a 'best fit' line
  # y = mlab.normpdf( bins, mu, sigma)
  # l = plt.plot(bins, y, 'r--', linewidth=1)

  plt.xlabel('CNN output probability')
  plt.ylabel('Number of jets')
  # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
  # plt.axis([40, 160, 0, 0.03])
  plt.grid(True)

  # plt.show()

  fig = plt.gcf()
  plot_FNAME = 'Hist_'+name+in_tuple+'.png'
  print('------------'*10)
  print('Hist plot name = ',plot_FNAME)
  print('------------'*10)
  plt.savefig(outprob_dir+plot_FNAME)

##-------------------------
# Plot the histogram
# make_hist(sig_prob,bg_prob, '_all_set')

# sys.exit()

##=============================================================================================
############       PLOT ROC CURVE
##=============================================================================================

#Make folder to save plots                                                                                                             
ROC_plots_dir = 'analysis/ROC/'
#os.system("rm -rf "+executedir)                                                                                                          
os.system("mkdir -p "+ROC_plots_dir)

ROC_plots_dir2 = 'analysis/ROC/'+str(in_tuple)+'/'
os.system("mkdir -p "+ROC_plots_dir2)


# Make ROC with area under the curve plot
def generate_results(y_test, y_score):
    #I modified from pos_label=1 to pos_label=0 because I found out that in my code signal is labeled as 0 and bg as 1
    fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=0, drop_intermediate=False)
    print('Thresholds[0:6] = \n',thresholds[:6])
    print('Thresholds lenght = \n',len(thresholds))
    print('fpr lenght',len(fpr))
    print('tpr lenght',len(tpr))

    print('------------'*10)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='red',label='Train epochs = '+str(epochs)+'\n ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot(fpr[2], tpr[2], color='red',
    #    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.xscale('log')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mistag Rate (False Positive Rate)')
    plt.ylabel('Signal Tag Efficiency (True Positive Rate)')
    plt.legend(loc="lower right")
    #plt.title('Receiver operating characteristic curve')
#     plt.show()
    plt.grid(True)
    fig = plt.gcf()
    label=''
    plot_FNAME = 'ROC_'+str(epochs)+'_'+in_tuple+label+'.png'
    plt.savefig(ROC_plots_dir2+plot_FNAME)

    ROC_FNAME = 'ROC_'+str(epochs)+'_'+in_tuple+label+'_Ntrain_'+str(Ntrain)+'.npy'
    np.save(ROC_plots_dir2+'fpr_'+str(sample_relative_size)+'_'+ROC_FNAME,fpr)
    np.save(ROC_plots_dir2+'tpr_'+str(sample_relative_size)+'_'+ROC_FNAME,tpr)
    print('ROC filename = {}'.format(ROC_plots_dir2+plot_FNAME))
    print('AUC =', np.float128(roc_auc))
    print('------------'*10)


generate_results(y_Test, out_prob)

# sys.exit()



##=============================================================================================
############       VISUALIZE CONVOLUTION RESULT 
##=============================================================================================

##--------------------------------------------------------------------------------------------- 
# Utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#pl.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))


##--------------------------------------------------------------------------------------------- 
# Get and plot the average of each intermediate convolutional layer 
##--------------------------------------------------------------------------------------------- 

#Split input image arrays into signal and background ones
x_test_sig=[]
x_test_bg=[]
# y_Test is the true value
n_sig=0
n_bg=0

for i_image in range(len(y_Test)):
  if y_Test[i_image]==0:
    x_test_sig.append(test_x[i_image])
    n_sig+=1
  elif y_Test[i_image]==1:
    x_test_bg.append(test_x[i_image])
    n_bg+=1


print('Lenght x_test signal = {} and number of signal samples = {}'.format(len(x_test_sig),n_sig))
print('Lenght x_test background {} and number of background samples = {}'.format(len(x_test_bg),n_bg))



# K.learning_phase() is a flag that indicates if the network is in training or
# predict phase. It allows layer (e.g. Dropout) to only be applied during training
inputs = [K.learning_phase()] + model.inputs

_convout1_f = K.function(inputs, [convout1.output])
_convout2_f = K.function(inputs, [convout2.output])
_convout3_f = K.function(inputs, [convout3.output])

# def convout1_f(X):
#     # The [0] is to disable the training phase flag
#     return _convout1_f([0] + [X])

# i = 3000
# Visualize the first layer of convolutions on an input image
# X = x_test[i:i+1]

# outlayer_plots_dir = 'analysis/outlayer_plots/'+name+'_epochs_'+str(epochs)+'/'
outlayer_plots_dir = 'analysis/outlayer_plots/'+'_'.join(in_tuple.split('_')[:-1])+'/'
#os.system("rm -rf "+executedir) 
os.system("mkdir -p "+outlayer_plots_dir)


def get_output_layer_avg(x_test,func,layer):
  
  if layer==1:
    avg_conv=np.zeros((32,17,17))
  elif layer==2:
    avg_conv=np.zeros((64,7,7))
  elif layer==3:
    avg_conv=np.zeros((64,3,3))
    
#   print("avg image shape = ", np.shape(avg_conv)) #create an array of zeros for the image

  for i_layer in range(len(x_test)):
    X = [x_test[i_layer]]
    # The [0] is to disable the training phase flag
    Conv = func([0] + [X])
#     print('Conv_array type  = ',type(Conv))
    Conv=np.asarray(Conv)
#     print('New type Conv_array type  = ',type(Conv))
#     print('Conv = \n',Conv[0:2])
#     print("First convolutional output layer shape before swapaxes = ", np.shape(Conv))
    Conv = np.squeeze(Conv) #Remove single-dimensional entries from the shape of an array.
    Conv=np.swapaxes(Conv,0,2) #Interchange two axes of an array.
    Conv=np.swapaxes(Conv,1,2)
    # print('Con[0]= \n',Conv[0])
#     print("First convolutional output layer shape after swap axes = ", np.shape(Conv))
#     print('-----------'*10)    
#     print('avg_conv=\n',avg_conv[0:2])
    avg_conv=avg_conv+Conv
  
  print("avg image shape after adding all images = ", np.shape(avg_conv)) #create an array of zeros for the image
  return avg_conv
  
  
  
def plot_output_layer_avg(func,n_im,layer,type):

  pl.figure(figsize=(15, 15))
  plt.axis('off')
  # pl.suptitle('convout1b')
  nice_imshow(pl.gca(), make_mosaic(func, n_im,8), cmap=cm.gnuplot)
  # return Conv
#   plt.show()
  fig = plt.gcf()
  plot_FNAME = 'avg_image_layer_'+str(layer)+'_'+str(type)+'_'+'_'.join(in_tuple.split('_')[4:-1])+'.png'
  # plot_FNAME = 'layer_'+str(layer)+'_img_'+str(i_im)+'_epochs_'+str(epochs)+'_'+in_tuple[:-4]+'.png'
  print('Saving average image for layer {} ...'.format(layer))
  print('-----------'*10)
  plt.savefig(outlayer_plots_dir+plot_FNAME)
  print('Output layer filename = {}'.format(outlayer_plots_dir+plot_FNAME))
  
  
# print('Name sig','_'.join(in_tuple.split('_')[4:-1]))

# avg_conv_array_sig=get_output_layer_avg(_convout2_f,2)
# avg_conv_array_sig1=get_output_layer_avg(x_test_sig,_convout1_f,1)
# avg_conv_array_bg1=get_output_layer_avg(x_test_bg,_convout1_f,1)
# avg_conv_array_sig2=get_output_layer_avg(x_test_sig,_convout2_f,2)
# avg_conv_array_bg2=get_output_layer_avg(x_test_bg,_convout2_f,2)
avg_conv_array_sig3=get_output_layer_avg(x_test_sig,_convout3_f,3)
avg_conv_array_bg3=get_output_layer_avg(x_test_bg,_convout3_f,3)

# plot_output_layer_avg(avg_conv_array_sig1,4,1,'tt')
# plot_output_layer_avg(avg_conv_array_bg1,4,1,'QCD')
# plot_output_layer_avg(avg_conv_array_sig2,8,2,'tt')
# plot_output_layer_avg(avg_conv_array_bg2,8,2,'QCD')
plot_output_layer_avg(avg_conv_array_sig3,8,3,'tt')
plot_output_layer_avg(avg_conv_array_bg3,8,3,'QCD')
 
# sys.exit()


##=============================================================================================
############       VISUALIZE WEIGHTS 
##=============================================================================================

W1 = model.layers[0].kernel
W2 = model.layers[2].kernel
W3 = model.layers[5].kernel

# all_W=[]
# for i_weight in range(3):
#   all_W.append(model.layers[i_weight].kernel)

# W is a tensorflow variable: type W =  <class 'tensorflow.python.ops.variables.Variable'>. We want to transform it to a numpy array to plot the weights
# print('type W1 = ',type(W1))
print('------------'*10)

import tensorflow as tf
# sess = tf.Session()
# # from keras import backend as K
# K.set_session(sess)
# weightmodel = tf.global_variables_initializer()

##---------------------------------------------------------------------------------------------
# Transform tensorflow Variable to a numpy array to plot the weights
def tf_to_np(weight):

  print('Type weight_array before opening a tensorflow session  = ',type(weight))
  sess = tf.Session()
  # from keras import backend as K
  K.set_session(sess)

  weightmodel = tf.global_variables_initializer()

  with sess:
    sess.run(weightmodel)
    weight_array=sess.run(weight)

  print('Type weight_array = ',type(weight_array))
  print('Shape weight_array before swapaxes = ',np.shape(weight_array))

  # weight_array=np.squeeze(weight_array)
  weight_array=np.swapaxes(weight_array,0,2)
  weight_array=np.swapaxes(weight_array,1,2)
  weight_array=np.asarray(weight_array)

  print('Shape weight_array after swapaxes = ',np.shape(weight_array))
  print('Shape weight_array after swapaxes[0] = ',np.shape(weight_array)[0])
  # print('Weight_aray = ',weight_array)

  return weight_array

# all_W_np=[]
# all_W_np.append(tf_to_np(W1))
# all_W_np.append(tf_to_np(W2))
# all_W_np.append(tf_to_np(W3))

##---------------------------------------------------------------------------------------------
# Plot the weights

weight_plots_dir = 'analysis/weight_plots/'+name+'_epochs_'+str(epochs)+'/'
#os.system("rm -rf "+executedir)                                                                                                          
os.system("mkdir -p "+weight_plots_dir)

# N_map=0

def plot_2nd_3d_layer(ww,N_out_layer,n_weight,n_row):

  wout=tf_to_np(ww)

  # if n_weight==2 or n_weight==3:
  wout=wout[N_out_layer]
  wout=np.swapaxes(wout,0,2)
  wout=np.swapaxes(wout,1,2)
  
  pl.figure(figsize=(15, 15))
  plt.axis('off')
  nice_imshow(pl.gca(), make_mosaic(wout, n_row, 8), cmap=cm.gnuplot)
  fig = plt.gcf()
  plot_FNAME = 'weights_'+str(n_weight)+'_epochs_'+str(epochs)+'_N_out_layer_'+str(N_out_layer)+'_'+in_tuple[:-4]+'.png'
#   plt.savefig(weight_plots_dir+plot_FNAME)
  print('Weights filename = {}'.format(weight_plots_dir+plot_FNAME))
  print('------------'*10)
 
  
for N_map in range(N_out_layer0):
  plot_2nd_3d_layer(W1,N_map,1,4)
  
# sys.exit()

for N_map in range(N_out_layer1):
  plot_2nd_3d_layer(W2,N_map,2,8)

for N_map in range(N_out_layer2):
  plot_2nd_3d_layer(W3,N_map,3,8)


##=============================================================================================
##=============================================================================================
##=============================================================================================
# Code execution time
print('-----------'*10)
print("Code execution time = %s minutes" % ((time.time() - start_time)/60))
print('-----------'*10)

  
