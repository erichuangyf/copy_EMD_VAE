# configs
# path to file

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

import os
import os.path as osp
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow.keras.backend as K

# from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
#import utils.VAE_model_tools_leakyrelu
#from utils.VAE_model_tools_leakyrelu import build_and_compile_annealing_vae, betaVAEModel, reset_metrics

import pandas
import matplotlib.pyplot as plt

import h5py
import pickle
from scipy.stats import gaussian_kde

import json

from pyjet import cluster, DTYPE_PTEPM
import tqdm


class VAE_sampler:

    def __init__(self, data_file_name, model_prefix, train_valid_split=800000, beta=0.001, lr=1e-4):
        self.x, self.y, self.vae, self.encoder, self.decoder, self.HT = None, None, None, None, None, None
        self._process_data(data_file_name, train_valid_split)
        #self._load_and_build_model(model_prefix, beta, lr)
        print("input data shape:{}".format(self.x.shape))
        print("output data shape:{}".format(self.y.shape))

    def run_analysis(self, weight_file, number_of_sampling=1, stop_index=None,
                     out_plot_prefix=None, save_sample_only=True, out_data_prefix=None,out_data_name=None,
                    additional_plot_config = None):
        # sample the jets using model checkpoint
        if stop_index:
            original_input, original_output = self.x[:stop_index], self.y[:stop_index]
        else:
            original_input, original_output = self.x, self.y
        print("vae input size: {}\nvalidation data size: {}".format(original_input.shape,original_output.shape))
        print("sampling data")
        outjets = self._generate_outjets(weight_file, number_of_sampling, original_input)
        # save the sampled data if requested
        if out_data_prefix:
            print("saving sampled data")
            if not save_sample_only:
                self._save_as_hdf5(outjets, out_data_prefix, save_name=out_data_name,original_output=original_output)
            else:
                self._save_as_hdf5(outjets, out_data_prefix, save_name=out_data_name,original_output=None)

        # generate plots
        plotting_method_name = \
            [method for method in dir(VAE_sampler) if method.startswith('_VAE_sampler__plots')]
        plotting_function = [eval("VAE_sampler."+func_name) for func_name in plotting_method_name]
        call_parameters = [original_output, outjets, out_plot_prefix]
        for count, func in enumerate(plotting_function):
            print("getting plot {} out of {}".format(count+1,len(plotting_function)))
            func(*call_parameters)
        return original_input, original_output, outjets, self.HT

    def _load_and_build_model(self, model_prefix, beta=0.001, lr=1e-4):
        vae_args_file = osp.join(model_prefix, "vae_args.dat")
        with open(vae_args_file, 'r') as f:
            vae_arg_dict = json.loads(f.read())
        self.vae, self.encoder, self.decoder = build_and_compile_annealing_vae(**vae_arg_dict)
        self.vae.beta.assign(beta)
        K.set_value(self.vae.optimizer.lr, lr)

    def _process_data(self, hf5_file, train_valid_split):
        """
        Process the data, and split into different datasets
        """
        df = pandas.read_hdf(hf5_file, stop=1000000)
        print(df.shape)
        print("Memory in GB:", sum(df.memory_usage(deep=True)) / (1024 ** 3) + sum(df.memory_usage(deep=True)) / (1024 ** 3))

        # Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi
        data = df.values.reshape((-1, 50, 4))

        # Normalize pTs so that HT = 1
        HT = np.sum(data[:, :, 0], axis=-1)
        self.HT = HT
        data[:, :, 0] = data[:, :, 0] / HT[:, None]

        # Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
        # Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
        # Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
        # Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
        sig_input = np.zeros((len(data), 50, 4))
        sig_input[:, :, :2] = data[:, :, :2]
        sig_input[:, :, 2] = np.cos(data[:, :, 2])
        sig_input[:, :, 3] = np.sin(data[:, :, 2])

        data_x = sig_input
        # Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
        data_y = data

        if train_valid_split and train_valid_split<data_x.shape[0]:
            self.x = data_x[train_valid_split:]
            self.y = data_y[train_valid_split:]
        else:
            self.x, self.y = data_x, data_y

    def _generate_outjets(self, weight_file, number_of_sampling, original_input):
        """Generate outjets from VAE model
            INPUT:
            weight file: weight file location for VAE model
            number_of_sampling: number of sampling
            OUTPUT:
            outs_jet: numpy array of outjets, shape: (number_of_sampling x data_inputsize)
        """
        self.vae.load_weights(weight_file)
        outs_jet = np.stack([self.vae.predict(original_input)[0] for j in range(number_of_sampling)])
        return outs_jet

    def _save_as_hdf5(self, sampled_jets, file_prefix, original_output=None, savename="jets.h5"):
        """Save data as hdf5 file
            original_jets: original input jets
            sampled_jets: sampled jets from VAE model
            file_prefix: output file prefix
            savename: output file name
        """
        hfile = h5py.File(osp.join(file_prefix, savename), 'w')
        if type(original_output) is not None:
            hfile.create_dataset('original_jets', data=original_output)
        hfile.create_dataset('sampled_jets', data=sampled_jets)
        hfile.close()

    @staticmethod
    def get_his(data, data_name, process_func=None, bins=None):
        for index, payload in enumerate(data):
            if process_func:
                payload = process_func(payload)
            if index == 0:
                n, b, _ = plt.hist(payload, label=data_name[index], alpha=0.5, histtype="step", density=True, bins=bins)
            else:
                plt.hist(payload, label=data_name[index], bins=b, alpha=0.5, histtype="step", density=True)

    # the plotting function starts here
    @staticmethod
    def __plots_constituent_eta(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        _bins = None
        if "bins" in kwargs:
            _bins = kwargs["bins"]
        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            VAE_sampler.get_his(data, data_name, lambda x: (x[:, :, 1]).flatten(), bins=_bins)
            # for index, payload in enumerate(data):
            #     plt.hist(payload[:, :, 1].flatten(), label=data_name[index], alpha=0.5, histtype="step",density=True)
        else:
            n, b, _ = plt.hist(ojs[:, :, 1].flatten(), label="Original", alpha=0.5,density=True)
            plt.hist(sjs[0][:, :, 1].flatten(), label="Sampled", bins=b, histtype="step", color="black",density=True)
        plt.xlabel("Constituent $\eta$")
        plt.legend()
        if save_plot:
            plt.savefig(osp.join(save_plot, "constituent_eta.png"))
        plt.show()

    @staticmethod
    def __plots_constituent_phi(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        _bins = None
        if "bins" in kwargs:
            _bins = kwargs["bins"]
        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            VAE_sampler.get_his(data, data_name, lambda x: (np.mod(x[:, :, 2], 2*np.pi) - np.pi).flatten(), bins=_bins)
            # for index, payload in enumerate(data):
            #     payload = np.mod(payload[:, :, 2], 2*np.pi) - np.pi
            #     plt.hist(payload.flatten(), label=data_name[index], alpha=0.5,histtype="step",density=True)
        else:
            phi = np.mod(ojs[:, :, 2], 2*np.pi) - np.pi
            n, b, _ = plt.hist(phi.flatten(), label="Original", alpha=0.5,density=True)
            plt.hist(sjs[0][:, :, 2].flatten(), label="Sampled", bins=b, histtype="step", color="black",density=True)
        plt.xlabel("Constituent $\phi$")
        plt.legend()
        if save_plot:
            plt.savefig(osp.join(save_plot, "constituent_phi.png"))
        plt.show()

    @staticmethod
    def __plots_constituent_pt(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        _bins = None
        if "bins" in kwargs:
            _bins = kwargs["bins"]
        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            VAE_sampler.get_his(data, data_name, lambda x: (x[:, :, 0]).flatten(), bins=_bins)
            # for index, payload in enumerate(data):
            #     plt.hist(payload[:, :, 0].flatten(), label=data_name[index], alpha=0.5, histtype="step",density=True)
        else:
            n, b, _ = plt.hist(ojs[:, :, 0].flatten(), label="Original", alpha=0.5,density=True)
            plt.hist(sjs[0][:, :, 0].flatten(), label="Sampled", bins=b, histtype="step", color="black",density=True)
        plt.xlabel("Constituent $p_T$ fraction")
        plt.yscale("log")
        plt.legend()
        if save_plot:
            plt.savefig(osp.join(save_plot, "constituent_pt.png"))
        plt.show()

    @staticmethod
    def event_mass(myinput):
        ms = []
        for i in range(len(myinput)):
            px = np.sum(myinput[i, :, 0].flatten() * np.cos(myinput[i, :, 2].flatten()))
            py = np.sum(myinput[i, :, 0].flatten() * np.sin(myinput[i, :, 2].flatten()))
            pz = np.sum(myinput[i, :, 0].flatten() * np.sinh(myinput[i, :, 1].flatten()))
            E = np.sum(myinput[i, :, 0].flatten() * np.cosh(myinput[i, :, 1].flatten()))
            ms += [(E ** 2 - px * px - py * py - pz * pz) ** 0.5]
        return np.array(ms)

    @staticmethod
    def __plots_event_mass(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        _bins = None
        if "bins" in kwargs:
            _bins = kwargs["bins"]
        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            VAE_sampler.get_his(data, data_name, VAE_sampler.event_mass, bins=_bins)
        else:
            event_mass_oj = VAE_sampler.event_mass(ojs)
            event_mass_sj = VAE_sampler.event_mass(sjs[0])
            n, b, _ = plt.hist(event_mass_oj, label="Original", alpha=0.5,density=True)
            plt.hist(event_mass_sj, label="Sampled", bins=b, histtype="step", color="black",density=True)
        plt.xlabel("Event Mass")
        plt.legend()
        if save_plot:
            plt.savefig(osp.join(save_plot, "event_mass.png"))
        plt.show()

    @staticmethod
    def MET(myinput):
        ms = []
        for i in range(len(myinput)):
            px = np.sum(myinput[i, :, 0].flatten() * np.cos(myinput[i, :, 2].flatten()))
            py = np.sum(myinput[i, :, 0].flatten() * np.sin(myinput[i, :, 2].flatten()))
            ms += [px ** 2 + py ** 2]
        return np.array(ms)

    @staticmethod
    def __plots_MET(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        _bins = None
        if "bins" in kwargs:
            _bins = kwargs["bins"]
        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            VAE_sampler.get_his(data, data_name, VAE_sampler.MET, bins=_bins)
        else:
            MET_oj = VAE_sampler.MET(ojs)
            MET_sj = VAE_sampler.MET(sjs[0])
            n, b, _ = plt.hist(MET_oj, label="Original", alpha=0.5,density=True)
            plt.hist(MET_sj, label="Sampled", bins=b, histtype="step", color="black",density=True)
        plt.xlabel("Missing Momentum")
        plt.legend()
        plt.yscale("log")
        if save_plot:
            plt.savefig(osp.join(save_plot, "MET.png"))
        plt.show()

    @staticmethod
    def jet_clustering(ojs, ptmin):
        print("clustering jets with paramert ptmin={}".format(ptmin))
        njets = []
        pTleadjet = []
        mleadjet = []
        for k in tqdm.tqdm(range(len(ojs))):
            pseudojets_input = np.zeros(50, dtype=DTYPE_PTEPM)
            for i in range(50):
                pseudojets_input[i]['pT'] = ojs[k, i, 0]
                pseudojets_input[i]['eta'] = ojs[k, i, 1]
                pseudojets_input[i]['phi'] = ojs[k, i, 2]
            sequence = cluster(pseudojets_input, R=0.4, p=-1)
            jets = sequence.inclusive_jets(ptmin=ptmin)  # 5 gev
            njets += [len(jets)]
            if (len(jets) > 0):
                pTleadjet += [jets[0].pt]
                mleadjet += [jets[0].mass]
        return njets, pTleadjet, mleadjet

    @staticmethod
    def __plots_clusterd_jets(ojs, sjs, save_plot, additional_signal=None, data_name=None, **kwargs):
        plot_label = ["n jets", "pT lead jet", "m lead jet"]
        _ptmin = 0.1
        _bins = None
        if "ptmin" in kwargs:
            _ptmin = kwargs["ptmin"]
            print("Alert: using customized argument: ptmin = {}".format(_ptmin))
        if "bins" in kwargs:
            _bins = kwargs["bins"]

        if additional_signal:
            data = [ojs, sjs]
            data.extend(additional_signal)
            clustering_info = []
            for d in data:
                clustering_info.append(VAE_sampler.jet_clustering(d,_ptmin))
            for i in range(3):
                plt.figure()
                itemized_info = [clustering_info[j][i] for j in range(len(data))]
                VAE_sampler.get_his(itemized_info, data_name, bins=_bins)
                plt.xlabel(plot_label[i])
                plt.legend()
                plt.yscale("log")
                if save_plot:
                    plt.savefig(osp.join(save_plot, plot_label[i]+".png"))
                plt.show()

        else:
            ojs_cluster = VAE_sampler.jet_clustering(ojs,_ptmin)
            sjs_cluster = VAE_sampler.jet_clustering(sjs[0],_ptmin)
            for i in range(3):
                plt.figure()
                n, b, _ = plt.hist(ojs_cluster[i], label="Original", alpha=0.5,density=True)
                plt.hist(sjs_cluster[i], label="Sampled", bins=b, histtype="step", color="black",density=True)
                plt.xlabel(plot_label[i])
                plt.legend()
                plt.yscale("log")
                if save_plot:
                    plt.savefig(osp.join(save_plot, plot_label[i]+".png"))
                plt.show()
