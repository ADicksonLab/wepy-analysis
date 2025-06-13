import sys
import os
import os.path as osp
import time
import numpy as np
#from scipy.stats import sem
#import pandas as pd
import pickle as pkl
import mdtraj as mdj
from wepy.hdf5 import WepyHDF5
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.analysis.contig_tree import ContigTree

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from deeptime.util.data import TimeLaggedDataset

def load_traj_field(wepy_h5, field_name):
    with wepy_h5:
        N_RUNS = wepy_h5.num_runs
        N_TRAJS = wepy_h5.num_run_trajs(0)
        N_CYCLES = wepy_h5.num_run_cycles(0)

        all_values = []
        for n_run in range(N_RUNS):
            traj_values = [np.asarray(wepy_h5.h5[f'runs/{n_run}/trajectories/{n_traj}/{field_name}']) for n_traj in range(N_TRAJS)]
            #traj_values = np.reshape(traj_values, (N_TRAJS, N_CYCLES, 1))
            all_values.append(traj_values)

    return np.asarray(all_values)

def load_weights_list(wepy_file, weight_path, out_file_name, save_file=True):

    """
    Load or compute trajectory weights for each run in a WepyHDF5 file.

    Parameters:
        wepy_file (WepyHDF5): Opened WepyHDF5 object.
        weight_path (str): Path to save or load weight files.
        out_file_name (str): Base name for saving/loading weight files.
        save_file (bool): Whether to save computed weights to disk.

    Returns:
        list: Nested list of weights per [run][walker, cycle].
    """


    wts = []

    with wepy_file:
        N_RUNS = wepy_file.num_runs
        for run in range(N_RUNS):
            save_file_name = osp.join(weight_path, f'{out_file_name}_{run}.npy')

            if not osp.exists(save_file_name):
                print(f"Building weights matrix...", flush=True)
                wts_per_run = []
                for walker in range(wepy_file.num_run_trajs(run)):
                    wts_per_run.append(np.asarray(wepy_file.h5[f'runs/{run}/trajectories/{walker}/weights/']))

                if save_file:
                    pkl.dump(wts_per_run, open(f'{save_file_name}','wb'))

            else:
                print(f"Loading weights for file {save_file_name}", flush=True)
                wts_per_run = pkl.load(open(save_file_name,'rb')) 
            
            wts.append(wts_per_run)

    return wts

def load_feature_file(feature_path, feature_name):

    out_file_name = f'{feature_name}.npy'
    save_file_name = osp.join(feature_path, out_file_name)

    if not osp.exists(save_file_name):
        print(f"File doesn't exist: {save_file_name}", flush=True)
        print(f"Passing", flush=True)
        pass

    else:
        print(f'Loading file: {save_file_name}', flush=True)
        arr_file = np.load(save_file_name)

        return arr_file

def load_feature_list(feature_path, feature_name, n_files):

    feature_arrays = []
    for cont in range(n_files):
        feature_arr = load_feature_file(feature_path=feature_path, 
                                        feature_name=f'{feature_name}_{cont}')
                
        if feature_arr is not None:
            feature_arrays.append(feature_arr)

    print(f'{len(feature_arrays)} files loaded for feature {feature_name}', flush=True)

    return feature_arrays

def load_features_dict(feature_path, features_list, n_files, reshape=True):

    print(f'Building features dictionary with features {features_list}', flush=True)
    features_dict = {}
    for n_feature, feature_name in enumerate(features_list):
        arrays = []
        for cont in range(n_files):
            arr = load_feature_file(feature_path = feature_path, 
                                    feature_name = f'{feature_name}_{cont}')
            
            if type(arr) is not type(None):
                if reshape:
                    if arr.ndim > 3:
                        new_shape = (arr.shape[0], arr.shape[1], np.prod(arr.shape[2:]))
                        arr = arr.reshape(new_shape)
                        arrays.append(arr)
                    else:
                        arrays.append(arr)
                else: 
                    arrays.append(arr)

        print(f'{len(arrays)} files loaded for feature {feature_name}', flush=True)
        features_dict["{0}".format(feature_name)] = arrays

    return features_dict

def build_features_matrix(feature_path, features_list, n_files, reshape=True):

    feat_dict = load_features_dict(feature_path=feature_path,  
                                    features_list = features_list, 
                                    n_files = n_files, 
                                    reshape=reshape,
                                    )

    print(f'Building a matrix with features: {feat_dict.keys()}', flush=True)
    features_all = []
    n_files = len(feat_dict[list(feat_dict.keys())[0]])
    for k in range(n_files):
        ## use it with reshape True
        stack = np.dstack([feat_dict[key][k] for key, values in feat_dict.items()])
        features_all.append(stack)

    return features_all, feat_dict

def normalization(wepy_file, data):
    print('Normalizing the data')
    data_conc = np.concatenate(data, axis=1)
    if data_conc.ndim < 3:
        data_conc = data_conc[..., np.newaxis]
        data_conc_r = data_conc.reshape(data_conc.shape[0]*data_conc.shape[1], data_conc.shape[2])
    else:
        data_conc_r = data_conc.reshape(data_conc.shape[0]*data_conc.shape[1], data_conc.shape[2])

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data_conc_r)
    norm = scaler.transform(data_conc_r)
    norm_r = norm.reshape(data_conc.shape[0], data_conc.shape[1], data_conc.shape[2])

    #reshape normalized features
    norm_data = []
    start_idx = 0
    with wepy_file:
        n_runs = wepy_file.num_runs
        n_walkers = wepy_file.num_run_trajs(0)
        for run_idx in range(n_runs):
            n_cycles = wepy_file.num_run_cycles(run_idx)
            end_idx = start_idx + n_cycles
            arr = norm_r[:, start_idx:end_idx, :]
            arr = np.squeeze(arr) 
            norm_data.append(arr)
            start_idx = end_idx

    return norm_data, scaler

def generate_random_dataset(features_list, output_path, n_train=450000):
    """
    Save a random subset of transformed features to a pickle file.
    """
    feat_name = osp.join(output_path, f"random_features.pkl")

    if osp.exists(feat_name):
        print('Loading random dataset...')
        with open(feat_name, 'rb') as f:
            rnd_list = pkl.load(f)
    else:
        print('Generating random dataset...')
        try:
            tmp = np.concatenate(features_list,axis=1)
        except:
            tmp = np.concatenate(features_list,axis=0)

        if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

        tmp_r = tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[-1])
        
        if tmp_r.shape[0] <= n_train:
            idx = np.arange(tmp_r.shape[0])
        else:
            idx = np.random.choice(np.arange(tmp_r.shape[0]), size=n_train, replace=False)
        rnd_list = tmp_r[idx]

        with open(feat_name, 'wb') as f:
            pkl.dump(rnd_list, f)

    return rnd_list

def build_time_lagged_dataset(wepy_h5, features_list, n_train, tau=1):

    window_length = int(tau) + 1
    with wepy_h5:
        print(f'Building contig tree..', flush=True)
        ct = ContigTree(wepy_h5=wepy_h5, decision_class=MultiCloneMergeDecision, boundary_condition_class=UnbindingBC)
        sw = ct.sliding_windows(window_length) 

    valid_sw = [i for i in range(len(sw)) if sw[i][0][0] == sw[i][1][0]] 
    if len(valid_sw) < n_train:
        sw_tica_train = [sw[i] for i in valid_sw]
        n_train = len(valid_sw)
    else:
        sw_idx_all = np.random.choice(valid_sw, size=n_train, replace=False)
        sw_tica_train = [sw[idx] for idx in sw_idx_all]
    
    tmp=[]
    for s in sw_tica_train:
        r,w,c=s[0]
        try:
            tmp.append(features_list[r][w,c])
        except:
            tmp.append(features_list[r][w,c,:])
    tmp_r = np.vstack(tmp)

    print("Generating time-lagged dataset", flush=True)
    n_dists = features_list[0].shape[-1]
    tl_data = np.zeros((2, n_train, n_dists))
    for i in range(n_train):
        run1,traj1,cycle1 = sw_tica_train[i][0]
        run2,traj2,cycle2 = sw_tica_train[i][-1]
        tl_data[0,i] = features_list[run1][traj1,cycle1]
        tl_data[1,i] = features_list[run2][traj2,cycle2]

    tld = TimeLaggedDataset(tl_data[0],tl_data[1])

    return tld, tmp_r
