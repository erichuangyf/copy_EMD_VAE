#!/usr/bin/env python
# coding: utf-8

# # Train on W-jets

# In[1]:


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow.keras as keras
import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import Flatten, Reshape, Lambda
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras import Model

import os
import os.path as osp
import sys

import numpy as np
#from scipy import linalg as LA

import matplotlib
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools
from utils.VAE_model_tools import build_and_compile_annealing_vae, betaVAEModel, reset_metrics

import pandas
import matplotlib.pyplot as plt

import h5py
import pickle


# In[2]:


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def ptetaphiE_to_Epxpypz(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = E
    newjets[:,:,1] = px
    newjets[:,:,2] = py
    newjets[:,:,3] = pz
    
    return newjets

def ptetaphiE_to_ptyphim(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    pz = pt * np.sinh(eta)
    y = 0.5*np.nan_to_num(np.log((E+pz)/(E-pz)))
    
    msqr = np.square(E)-np.square(pt)-np.square(pz)
    msqr[np.abs(msqr) < 1e-6] = 0
    m = np.sqrt(msqr)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = y
    newjets[:,:,2] = phi
    newjets[:,:,3] = m
    
    return newjets
    
def ptyphim_to_ptetaphiE(jets):
    
    pt = jets[:,:,0]
    y = jets[:,:,1]
    phi = jets[:,:,2]
    m = jets[:,:,3]
    
    eta = np.nan_to_num(np.arcsinh(np.sinh(y)*np.sqrt(1+np.square(m/pt))))
    pz = pt * np.sinh(eta)
    E = np.sqrt(np.square(pz)+np.square(pt)+np.square(m))
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = eta
    newjets[:,:,2] = phi
    newjets[:,:,3] = E
    
    return newjets
    
def center_jets_ptetaphiE(jets):
    cartesian_jets = ptetaphiE_to_Epxpypz(jets)
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)
    
    sumjet_phi = np.arctan2(sumjet_cartesian[:,2],sumjet_cartesian[:,1])
    sumjet_y = 0.5*np.log((sumjet_cartesian[:,0] + sumjet_cartesian[:,-1])/(sumjet_cartesian[:,0] - sumjet_cartesian[:,-1]))
    
    ptyphim_jets = ptetaphiE_to_ptyphim(jets)
    #print(ptyphim_jets[:3,:,:])
    
    transformed_jets = np.copy(ptyphim_jets)
    transformed_jets[:,:,1] = ptyphim_jets[:,:,1] - sumjet_y[:,None]
    transformed_jets[:,:,2] = ptyphim_jets[:,:,2] - sumjet_phi[:,None]
    transformed_jets[:,:,2] = transformed_jets[:,:,2] + np.pi
    transformed_jets[:,:,2] = np.mod(transformed_jets[:,:,2],2*np.pi)
    transformed_jets[:,:,2] = transformed_jets[:,:,2] - np.pi

    transformed_jets[transformed_jets[:,:,0] == 0] = 0
    
    newjets = ptyphim_to_ptetaphiE(transformed_jets)
    return newjets
    


# ## Load and preprocess train/val data

# In[3]:


# path to file
fn =  '/global/home/users/yifengh3/data/monoW-data-3.h5'
# fn =  '/media/jcollins/MAGIC!/monoW-data-3.h5'

# Option 1: Load everything into memory
df = pandas.read_hdf(fn,stop=1000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))


# In[4]:


# Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi, E.
data = df.values.reshape((-1,50,4))

# Normalize pTs so that HT = 1
HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

# Center jet (optional)
data = center_jets_ptetaphiE(data)

# Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
# Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
# Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
# Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
sig_input = np.zeros((len(data),50,5))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)


data_x = sig_input
# Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
data_y = data[:,:,:3]


train_x = data_x[:500000]
train_y = data_y[:500000]
valid_x = data_x[500000:600000]
valid_y = data_y[500000:600000]


# In[5]:


output_dir = './output/'

experiment_name = 'W-test_2'
train_output_dir = create_dir(osp.join(output_dir, experiment_name))
vae, encoder, decoder = build_and_compile_annealing_vae(optimizer=keras.optimizers.Adam(lr=0.001,clipnorm=0.1),
                                    encoder_conv_layers = [1024,1024,1028,1024],
                                    dense_size = [1028,1028,1028,512],
                                    decoder = [2048,2048,1028,512,512],
                                    numItermaxinner = 40,   # EMD approximation params
                                    numIter=10,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    stopThr=1e-3,
                                    num_inputs=5,           # Size of x (e.g. pT, eta, sin, cos, log E)
                                    num_particles_in=50)    # Num particles per event.

batch_size=100
save_period=2

reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0)
modelcheckpoint = keras.callbacks.ModelCheckpoint(train_output_dir + '/model_weights_{epoch:02d}.hdf5', save_freq = save_period*5000, save_weights_only=True)
reset_metrics_inst = reset_metrics()

callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
            reduceLR,
            modelcheckpoint,
            reset_metrics_inst]


# Need to train on at least one example before model params can be loaded for annoying reasons.

history = vae.fit(x=train_x[:10], y=train_y[:10], batch_size=batch_size,
                epochs=1,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
                validation_data = (valid_x[:10],valid_y[:10]),
                callbacks = callbacks
              )


# In[6]:


# init_epoch = 3957
# beta = 1.6e-4
# # vae.load_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + str(beta) + '.hdf5')
# vae.load_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')


# In[7]:


# # init_epoch = 544
# init_epoch = 0
# steps_per_epoch = 1000
# save_period = 10


# reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
# earlystop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )

# #beta_array = np.logspace(-3,np.log10(4e-2),5)
# beta_array = np.concatenate([np.logspace(-3,np.log10(4e-2),5),np.logspace(-3,np.log10(4e-2),5)[::-1]])

# for beta in beta_array:
#     modelcheckpoint = keras.callbacks.ModelCheckpoint(train_output_dir + '/model_weights_{epoch:02d}_' + "{:.1e}".format(beta) + '.hdf5', save_freq = save_period*steps_per_epoch, save_weights_only=True)
#     callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
#             reduceLR,earlystop,
#             modelcheckpoint,
#             reset_metrics_inst]
#     vae.beta.assign(beta)
#     K.set_value(vae.optimizer.lr,3e-5)
    
#     my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
#                 epochs=10000,verbose=1,
#                 validation_data = (valid_x[:200*batch_size],valid_y[:200*batch_size]),
#                 callbacks = callbacks,
#                 initial_epoch=init_epoch,
#                 steps_per_epoch = steps_per_epoch
#               )
#     init_epoch = my_history.epoch[-1]
#     vae.save_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')


# In[9]:


# %ls output/W-test/


# In[10]:


# init_epoch = 428
# vae.load_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')


# In[ ]:


init_epoch = 0
steps_per_epoch = 1000
save_period = 10


# reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
# earlystop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )

reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

for beta in np.concatenate((np.logspace(-4,np.log10(0.7),10),
               np.logspace(np.log10(0.7),-5,10)[1:],
                np.logspace(-5,np.log10(0.7),10)[1:])):
    modelcheckpoint = keras.callbacks.ModelCheckpoint(train_output_dir + '/model_weights_{epoch:02d}.hdf5', save_freq = save_period*steps_per_epoch, save_weights_only=True)
    callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
            reduceLR,earlystop,
            modelcheckpoint,
            reset_metrics_inst]
    vae.beta.assign(beta)
    K.set_value(vae.optimizer.lr,3e-5)
    
    my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
                epochs=10000,verbose=1,
                validation_data = (valid_x[:200*batch_size],valid_y[:200*batch_size]),
                callbacks = callbacks,
                initial_epoch=init_epoch,
                steps_per_epoch = steps_per_epoch
              )
    init_epoch = my_history.epoch[-1]
    vae.save_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')


