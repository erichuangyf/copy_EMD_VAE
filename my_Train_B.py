#!/usr/bin/env python
# coding: utf-8

# # Train on B-jets

# In[1]:

output_dir = '/global/home/users/yifengh3/VAE/B_results'
experiment_name = 'method2_beta2'


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

import os
import os.path as osp
import sys

import numpy as np
#from scipy import linalg as LA

import matplotlib
import matplotlib.pyplot as plt


from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools_leakyrelu
from utils.VAE_model_tools_leakyrelu import build_and_compile_annealing_vae, betaVAEModel, reset_metrics

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
fn =  '/global/home/users/yifengh3/VAE/data/B_background.h5'

# Option 1: Load everything into memory
df = pandas.read_hdf(fn,stop=1000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))


# In[4]:


# Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi
data = df.values.reshape((-1,50,3))

# Normalize pTs so that HT = 1
HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
# Energy column DNE here, so dont need to normalize the data
# data[:,:,-1] = data[:,:,-1]/HT[:,None]

# Center jet (optional)
# cant center the jets since E missing here
# data = center_jets_ptetaphiE(data)

# Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
# Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
# Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
# Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
sig_input = np.zeros((len(data),50,4))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
# no input from energy for B jets
# sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)


data_x = sig_input
# Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
data_y = data


train_x = data_x[:800000]
train_y = data_y[:800000]
valid_x = data_x[800000:]
valid_y = data_y[800000:]


# In[5]:


import json

train_output_dir = create_dir(osp.join(output_dir, experiment_name))
vae_args_file = osp.join(train_output_dir,"vae_args.dat")
vae_arg_dict = None

# if the arg file exists, load model config
if os.path.isfile(vae_args_file):
    with open(vae_args_file,'r') as f:
          vae_arg_dict = json.loads(f.read())
else:
    # otherwize use the initial argument dict
    vae_arg_dict = {"encoder_conv_layers" : [1024,1024,1024,1024],
                    "dense_size" :[1024,1024,1024,1024],
                    "decoder_sizes" : [1024,1024,1024,1024,1024],
                    "numItermaxinner" : 20,   # EMD approximation params
                    "numIter":10,
                    "reg_init" : 1.,
                    "reg_final" : 0.01,
                    "stopThr":1e-3,
                    "num_inputs":4,           # Size of x (e.g. pT, eta, sin, cos, log E)
                    "num_particles_in":50}
    with open(vae_args_file,'w') as file:
      file.write(json.dumps(vae_arg_dict))
    
vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)


# In[6]:


beta_set = np.logspace(-5,1,25)[:-5]
betas = beta_set
for i in range(0,16,2):
  betas = np.append(betas, beta_set[-1-7-i:-1-i])
betas = np.append(betas, beta_set)
#plt.plot(betas)
#plt.semilogy()
#plt.show()


# In[7]:


init_epoch = 0
steps_per_epoch = 1000
batch_size=100
save_period=10

# define some directory so the model file will not appears everywhere
checkpoint_dir = create_dir(osp.join(train_output_dir, "checkpoint"))
end_beta_checkpoint = create_dir(osp.join(train_output_dir, "end_beta_checkpoint"))
print("end_beta checkpoint will now be {}".format(end_beta_checkpoint))

modelcheckpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir + '/model_weights_{epoch:02d}.hdf5', save_freq = save_period*5000, save_weights_only=True)

reset_metrics_inst = reset_metrics()

reduceLR = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=np.sqrt(0.1), 
    patience=5, verbose=1, mode='auto', 
    min_delta=1e-4, cooldown=0, min_lr=1e-8)


callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
            reduceLR,
            modelcheckpoint,
            reset_metrics_inst]

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

for beta in betas:
    vae.beta.assign(beta)
    K.set_value(vae.optimizer.lr,3e-5)
    
    callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
            reduceLR,earlystop,
            modelcheckpoint,
            reset_metrics_inst]
    
    my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
                epochs=10000,verbose=1,
                validation_data = (valid_x[:200*batch_size],valid_y[:200*batch_size]),
                callbacks = callbacks,
                initial_epoch=init_epoch,
                steps_per_epoch = steps_per_epoch
              )
    
    init_epoch = my_history.epoch[-1]
    
    print("saveing model to {}".format(osp.join(end_beta_checkpoint,'model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')))
    
    vae.save_weights(osp.join(end_beta_checkpoint,'model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5'))


# In[ ]:




