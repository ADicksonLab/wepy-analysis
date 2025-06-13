import os.path as osp
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
from csnanalysis.csn import CSN
from csnanalysis.matrix import *


def cluster_data(dataset, outpath, algorithm='kmeans', n_clusters=1000, random_state=None):
    """
    Perform clustering on the features and save or load the clustering model.

    """

    model_name = osp.join(outpath, f'{algorithm}_rs{random_state}.pkl')
    if osp.exists(model_name):
        print("Reading existing clustering model..")
        with open(model_name, 'rb') as f:
            clustering_model = pkl.load(f)
    else:
        t1 = time.time()
        print(f"Fitting {algorithm} model..")

        if algorithm == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state)
            clustering_model.fit(dataset)

        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        with open(model_name, 'wb') as f:
            pkl.dump(clustering_model, f)

        t2 = time.time()
        print(f'Time taken for clustering with {algorithm}: {t2-t1} seconds\n')

    return clustering_model

def generate_cluster_labels(outpath, features_list, estimator, kmeans):
    print('Generating cluster labels...', flush=True)

    cluster_labels = []
    tica_values = []
    for run_idx in range(len(features_list)):
        print(f"Run {run_idx} of {len(features_list)}..", flush=True)

        ncycles = features_list[run_idx].shape[1]
        labels = np.zeros((features_list[run_idx].shape[0],ncycles),dtype=int)
        vals = np.zeros((features_list[run_idx].shape[0], ncycles, estimator.dim),dtype=int)
        for walker in range(features_list[run_idx].shape[0]):
            tica_vals = [estimator.transform(f) for f in features_list[run_idx][walker]]
            tmp_labels = kmeans.predict(np.array(tica_vals))
            labels[walker] = tmp_labels
            vals[walker] = tica_vals

        cluster_labels.append(labels)
        tica_values.append(vals)

    with open(osp.join(outpath, f'cluster_labels.pkl'),'wb') as f:
        pkl.dump(cluster_labels,f)

    with open(osp.join(outpath, f'tica_values.pkl'),'wb') as f:
        pkl.dump(tica_values,f)

    return cluster_labels, tica_values

def build_or_load_properties(outpath, wepy_file, n_clusters, all_labels, wts, lig_rmsd, unb_min_dist, rmsd_cutoff_bound=0.3, mind_cutoff_unbound=0.5, n_eval_points=5000):
    """
    Build or load properties for each cluster.

    Parameters:
        outpath  (Path): Output folder to save or load properties.
        n_clusters (int): Total number of clusters.
        wepy_file: WepyHDF5 object with trajectory data.
        all_labels (Nested list [run][walker][cycle]): Cluster labels for all WE data. 
        wts (Nested list [run][walker, cycle]): WE weight array.
        lig_rmsd (Nested list [run][walker][cycle]): Ligand RMSD data to define bound states.
        unb_min_dist (Nested list [run][walker][cycle]): Min distances to define unbound states.
        rmsd_cutoff_bound (int): RMSD threshold for bound states. 
        mind_cutoff_unbound (int): Distance threshold for unbound states.
        n_eval_points (int): Max number of points to sample per cluster. 

    Returns:
        Dictionary with computed properties per cluster.

    """
    prop_path = osp.join(outpath, 'properties.pkl')
    if osp.exists(prop_path):
        print("Load in properties")
        with open(prop_path, 'rb') as f:
            properties = pkl.load(f)
    else:
        print('Building properties with dcd...', flush=True)
        properties = {}
        trajs = {}
        tot_wts = np.zeros((n_clusters))
        av_rmsds = np.zeros((n_clusters))
        av_unb = np.zeros((n_clusters))
        min_rmsds = np.zeros((n_clusters))
        max_rmsds = np.zeros((n_clusters))
        nav_rmsds = np.zeros((n_clusters))
        nav_unb = np.zeros((n_clusters))
        max_unb_min_dists = np.zeros((n_clusters))
        min_unb_min_dists = np.zeros((n_clusters))   
        wts = np.zeros((n_clusters))
        free_energy_max = 100.

        field_list = [['weights'], 'unb_min_dist', 'lig_rmsd', 'sub_traces']

        trajs = {}
        wts = np.zeros((n_clusters))
        with wepy_file:
            for clust in range(n_clusters):
                if (clust % 100) == 0:
                    print(f"Determining node properties for cluster {clust}")
                trace = []
                for run, clust_idxs in enumerate(all_labels):
                    walkers, cycles = np.where(clust_idxs == clust)
                    n = len(walkers)
                    if n > 0:
                        runs = np.ones((n), dtype=int) * run
                        trace += list(zip(runs, walkers, cycles))
                if len(trace) > 0:
                    if len(trace) > n_eval_points:
                        sub_trace = [trace[i] for i in np.random.choice(len(trace), n_eval_points)]
                    else:
                        print("cluster", clust, "has", len(trace), "points")
                        sub_trace = trace
                        
                    props = wepy_file.get_trace_fields(sub_trace, field_list[0])
                    min_dist_arr = []
                    lig_rmsd_arr = []
                    phi_values_arr = []
                    for i in range(len(sub_trace)):
                        rmsd = lig_rmsd[sub_trace[i][0]][sub_trace[i][1]][sub_trace[i][2]]
                        lig_rmsd_arr.append([rmsd])
                        dist = unb_min_dist[sub_trace[i][0]][sub_trace[i][1]][sub_trace[i][2]]
                        min_dist_arr.append([dist])

                        props[field_list[1]] = np.array(min_dist_arr)
                        props[field_list[2]] = np.array(lig_rmsd_arr)

                wts[clust] = props['weights'][:,0].mean()
                av_rmsds[clust] += props['lig_rmsd'].sum()
                min_rmsd = props['lig_rmsd'].min()
                max_rmsd = props['lig_rmsd'].max()
                max_unb = props['unb_min_dist'].max()
                min_unb = props['unb_min_dist'].min()
                av_unb[clust] = props['unb_min_dist'].sum()
                if max_unb > max_unb_min_dists[clust]:
                    max_unb_min_dists[clust] = max_unb
                if min_unb < min_unb_min_dists[clust] or min_unb_min_dists[clust] == 0:
                    min_unb_min_dists[clust] = min_unb
                if max_rmsd > max_rmsds[clust]:
                    max_rmsds[clust] = max_rmsd
                if min_rmsd < min_rmsds[clust] or min_rmsds[clust] == 0:
                    min_rmsds[clust] = min_rmsd

                nav_rmsds[clust] += len(props['lig_rmsd'])
                nav_unb[clust] += len(props['unb_min_dist'])
                trajs[clust] = sub_trace

        tot_wts += wts

        # get free energy from weights
        fe = -np.log(tot_wts,where=tot_wts>0)
        fe -= fe[np.where(tot_wts>0)].min()
        fe[np.where(tot_wts==0)] = free_energy_max
        properties['fe'] = fe
        # add total weight to csn
        properties['tot_weight'] = tot_wts/tot_wts.sum()
        # add averaged attributes to csn
        av_rmsds /= nav_rmsds
        av_unb /= nav_unb

        properties['rmsd_range'] = max_rmsds - min_rmsds
        properties['unb_dist_range'] = max_unb_min_dists - min_unb_min_dists
        properties['lig_rmsd'] = av_rmsds
        properties['unb_min_dist_max'] = max_unb_min_dists
        properties['unb_min_dist_avg'] = av_unb
        # determine bound and unbound states
        bound = av_rmsds < rmsd_cutoff_bound
        # unbound = max_unb_min_dists > mind_cutoff_unbound
        unbound = av_unb > mind_cutoff_unbound
        properties['is_bound'] = np.array(bound,dtype=int)
        properties['is_unbound'] = np.array(unbound,dtype=int)
        properties['sub_traces'] = trajs

        with open(prop_path_dcd,'wb') as f:
            pkl.dump(properties,f)
        print(f"Saved properties to {prop_path}")

    return properties

def build_count_matrix(outpath, wepy_file, n_clusters, all_labels, wts, tau=1):
    """
    Build or load the count matrix for the given lags.
    """
    counts_name = osp.join(outpath, f'counts_matrix.pkl')

    if osp.exists(counts_name):
        print("Reading count matrix")
        with open(counts_name, 'rb') as f:
            all_counts = pkl.load(f)
    else:
        print('Building count matrix')
        sw_sets = {}
        all_counts = {}

        window_length = tau + 1
        with wepy_file:
            contig_tree = ContigTree(wepy_h5=wepy_file, decision_class=MultiCloneMergeDecision, boundary_condition_class=UnbindingBC)
            sw_sets[tau] = contig_tree.sliding_windows(window_length)  #(run_idx, traj_idx, cycle_idx)

        counts = np.zeros((n_clusters,n_clusters))
        for s in sw_sets[tau]:
            if s[0][0] == s[-1][0]:
                # s[0] (run, walker, cycle at time 0)
                # s[1] (run, walker, cycle at time t)
                c0 = all_labels[s[0][0]][s[0][1], s[0][2]]
                ct = all_labels[s[-1][0]][s[-1][1], s[-1][2]]
                counts[c0, ct] += wts[s[-1][0]][s[-1][1]][s[-1][2]]
        all_counts[tau] = counts

        with open(counts_name, 'wb') as f:
            pkl.dump(all_counts, f)
        print(f"Saved count matrices to {counts_name}")

    return all_counts


def calculate_mfpt_and_fptd(outpath, properties, all_counts, tau=1):
    """
    Calculate MFPTs, FPTDs, and save the network with additional attributes for visualization.

    """
    mfpt_path = osp.join(outpath, 'mfpts.pkl')
    fptds_path = osp.join(outpath, 'fptds.pkl')
    comm_path = osp.join(outpath, 'committors.pkl')

    if osp.exists(mfpt_path):
        print("Load in mfpts")
        with open(mfpt_path, 'rb') as f:
            mfpts = pkl.load(f)
        with open(fptds_path, 'rb') as f:
            fptds = pkl.load(f)
        with open(comm_path, 'rb') as f:
            committors = pkl.load(f)
    else:
        print('Calculating mfpts and fptds..')
        mfpts = {}
        fptds = {}
        committors = {}

        lig_csn = CSN(all_counts[tau].T) #CSN expects [to][from]
        lig_csn.trim()
        unb = np.where(properties['is_unbound'] == 1)[0]
        b = np.where(properties['is_bound'] == 1)[0]

        conflict = False
        for u in unb:
            if u in b:
                conflict = True
        print("unbound states:", unb)
        print("bound states:", b)
        if conflict:
            print("Warning: one or more states is both bound and unbound. MFPT set to zero")
            mfpts[tau] = 0
        elif len(unb) > 0 and len(b) > 0:
            mfpt, fptds[tau] = lig_csn.calc_mfpt(unb, maxsteps=10000, tol=1e-2, sources=b)
            if fptds[tau].shape[1] == 10000:
                unb_frac = fptds[tau][0].sum()
                last_1k = fptds[tau][0][-1000:].sum()
                print(f"Max steps reached.  Unbound fraction = {unb_frac} (last 1000: {last_1k})")
                mfpts[tau] = mfpt * tau
            else:
                mfpts[tau] = mfpt * tau

            mfpts['unbound states'] = unb
            mfpts['bound states'] = b
            print("Determining committors..")
            committors = lig_csn.calc_committors([b, unb], maxstep=200)
        else:
            print("Warning: bound and/or unbound states not present. MFPT set to None")
            mfpts[tau] = None

        for p in properties.keys():
            lig_csn.add_attr(p, properties[p])

        nx.write_gexf(lig_csn.graph,osp.join(outpath,  'network.gexf'))

        with open(mfpt_path, 'wb') as f:
            pkl.dump(mfpts, f)
        with open(fptds_path, 'wb') as f:
            pkl.dump(fptds, f)
        with open(comm_path, 'wb') as f:
            pkl.dump(committors, f)
        print(f"Saved mfpts and fptds to {mfpt_path} and {fptds_path}")

    return mfpts, fptds, committors

