import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from wepy.hdf5 import WepyHDF5
from deeptime.util.data import TimeLaggedDataset
from deeptime.decomposition import TICA
from sklearn.cluster import KMeans
from csnanalysis.csn import CSN
from csnanalysis.matrix import *

sys.path.insert(0, str(Path('.') / 'src'))
from wepyanalysis.featurization.distance import DistanceFeatureGenerator
from wepyanalysis.datasets.generator import load_weights_list, load_feature_list, build_features_matrix, normalization, generate_random_dataset, build_time_lagged_dataset
from wepyanalysis.msm.builder import cluster_data, generate_cluster_labels, build_or_load_properties, build_count_matrix, calculate_mfpt_and_fptd
from wepyanalysis.hdf5 import link_wepy_runs

#### This code provides an analysis framework to build MSM from WE data. 
#### It assumes the unbinding simulations are done and builds evertything after that. 
#### Check simulations folder to find an example script to how to run wepy simulations. 
#### Refer to our documentation for detailed description of how to prepare your data and run simulations. 
#### (https://adicksonlab.github.io/wepy/index.html)


#### Define required parameters
# Full path to the folder containing system files (.pdb, .rst, etc.)
base_path = Path('.')
# Folder path to save output files 
output_path = Path('.')
# Starting structure of the system
pdb = mdj.load_pdb(osp.join(base_path, 'step3_charmm2omm.pdb'))
# MDTraj selection string for ligand CA atoms
lig_idxs = 'segname PROC and name CA'
# MDTraj selection string for receptor atoms. This is a heterodimer so we'll provide a list including 
# the string selection separately for each protein. If it's a monomer, it is possible to pass a string or list of a string.
receptor_idxs = ['segname PROA and name CA', 'segname PROB and name CA']

# As an example, we will work with 3 hdf5 files, which are continuation of another based on numbering.
input_files = ['wepy.results0.h5', 'wepy.results1.h5', 'wepy.results2.h5']
wepy_files = [WepyHDF5(osp.join(base_path, inp_file), mode='r') for inp_file in input_files]

# ---------------------------------------------
#   STEP 1. FEATURE GENERATION
#----------------------------------------------
#### Generate distance features from WE data using DistanceFeatureGenerator and save as np array

for i, wepy_file in enumerate(wepy_files):
    feature_generator = DistanceFeatureGenerator(wepy_h5 = wepy_file, pdb = pdb, lig_idxs = lig_idxs, prot_idxs = receptor_idxs)
    
    # calculate RMSD of ligand CA atoms
    ligand_rmsd = feature_generator.call_compute_ligand_rmsd(out_file_name = f'rmsds_ligand_ca_{i}.npy',
                                                            out_folder_name= output_path, 
                                                            save_file=True)

    # calculate beta turn distance between two atoms
    ALA36_HN_idx = 3038 # Atom indicies from pdb
    GLY33_oxy_idx  = 3003
    pair_list_beta_turn = [(ALA36_HN_idx, GLY33_oxy_idx)]
    pair_dist = feature_generator.call_compute_pair_distances(pairs_list = pair_list_beta_turn,
                                                            out_file_name = f'beta_turn_{i}.npy',
                                                            out_folder_name= output_path, 
                                                            save_file=True)

    # calculate pairwise distances between two selected groups
    clr_bs_ca_idxs = pdb.top.select('segname PROB and ((residue 2071 2072 2114 2116 2117 2119) or (residue 2121 to 2125) or (residue 2128 to 2130) or (residue 2092 to 2095)) and name CA')
    ligand_ca_idxs = pdb.top.select('segname PROC and name CA')
    pair_list_dist_clr_ligand_ca = [(clr_bs_ca_idxs, ligand_ca_idxs)]
    pairwise_dist = feature_generator.call_compute_pairwise_distances(pairs_list = pair_list_dist_clr_ligand_ca,
                                                            out_file_name = f'dist_bs_lig_ca_{i}.npy',
                                                            out_folder_name= output_path, 
                                                            save_file=True)

    # calculate minimum distance between binding pocket of CLR and ligand
    clr_bs_idxs = pdb.top.select('segname PROB and (residue 2071 2072 2114 2116 2117 2119) or (residue 2121 to 2125) or (residue 2128 to 2130) or (residue 2092 to 2095)')
    ligand_idxs = [p.top.select('segname PROC') for p in PDBS] #CGRP and ssCGRP
    pair_list_min_dist_clr_ligand = [(clr_bs_idxs, ligand_idxs)] 
    min_dist = feature_generator.call_compute_min_dist(group1 = pair_list_min_dist_clr_ligand[0][0],
                                                        group2 = pair_list_min_dist_clr_ligand[0][1] 
                                                        out_file_name = f'min_dist_clr_ligand_{i}.npy',
                                                        out_folder_name= output_path, 
                                                        save_file=True)

# --------------------------------------------
#   STEP 2. DATASET GENERATION
#---------------------------------------------

## To make calculations easier, link continuation file to first run.
## ! Do not repeat this if the files are already linked since it will be keep adding it to the file
link_wepy_runs(base_path, input_files)

## Ue pairwise distances features as our main feature and generate a matrix with all files.
## Here we give the name of the files without number and extension along with number of files as an input
## If you want to combine different features together simply provide all of their names in the same format in a list

features_all, features_dict = build_features_matrix(base_path, ['dist_bs_lig_ca'], 
                                                    n_files=3, reshape=True)

# Normalize the features
features_all_norm, _ = normalization(wepy_files[0], features_all)

# Load RMSD and minimum distance features to use them to identify bound and unbound states. 
lig_rmsds = load_feature_list(feature_path=base_path, feature_name='rmsds_ligand_ca', n_files=3)
unb_min_dists = load_feature_list(feature_path=base_path, feature_name='min_dist_clr_ligand', n_files=3)

# Load WE weights. Reading weights from hdf5 files every time for a calculation can be costly. 
# Because of that this function saves it as a .npy file and reads from these files after first time. 
weights = load_weights_list(wepy_file=wepy_files[0], weight_path=base_path, 
                            out_file_name='wts', save_file=True)

# Generate a random dataset to perform tica training and clustering since using all data might not be suitable. 
# If the data size is smaller than the number selected, this function will include all data.
rnd_list = generate_random_dataset(features_list=features_all_norm, output_path=base_path, n_train=450000)

# Build time-lagged data using sliding window function in Wepy
tld, fit_dataset = build_time_lagged_dataset(wepy_h5=wepy_files[0], features_list=features_all_norm, n_train=450000, tau=1)

# -------------------------------------------------
#   STEP 3. BUILDING MARKOV STATE MODELS (MSMs)
#--------------------------------------------------

# Train tICA to lower dimensions
estimator = TICA(dim=3, lagtime=1).fit(tld)
est_list = estimator.transform(fit_dataset)

# Cluster transformed dataset using KMeans algorithm
kmeans = cluster_data(dataset=est_list, outpath=base_path, algorithm='kmeans', n_clusters=1000, random_state=3)

# Generate cluster labels and tica values for all WE data
cluster_labels, tica_values = generate_cluster_labels(outpath=base_path, features_list=features_all_norm, 
                                                    estimator=estimator, kmeans=kmeans)

# Assign properties to each cluster
properties = build_or_load_properties(outpath=base_path, wepy_file=wepy_files[0], n_clusters=1000, 
                                    all_labels=cluster_labels, wts=weights, lig_rmsd=lig_rmsds, 
                                    unb_min_dist=unb_min_dists, rmsd_cutoff_bound=0.3, 
                                    mind_cutoff_unbound=0.8, n_eval_points=5000)

# Build count matrix
all_counts = build_count_matrix(outpath=base_path, wepy_file=wepy_files[0], n_clusters=1000, 
                                all_labels=cluster_labels, wts=weights, tau=1)

# Calculate Conformational Space Network, committors and unbidning rates
mfpts, fptds, committors = calculate_mfpt_and_fptd(outpath=base_path, properties=properties, all_counts=all_counts, tau=1)
