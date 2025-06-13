import os
import os.path as osp
import time 
from pathlib import Path

import mdtraj as mdj
import numpy as np
from numpy import linalg as LA
import pickle as pkl
from sklearn.neighbors import KDTree

#wepy modules-h5
from wepy.hdf5 import WepyHDF5
from wepy.util.util import traj_box_vectors_to_lengths_angles

#geomm-mdtraj-numpy-pandas-itertools
from geomm.rmsd import calc_rmsd
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.centering import apply_rectangular_pbcs

def _load_pdb(pdb):

    """
    If `pdb` is already an mdtraj.Trajectory, return it.
    If it's a path (str, Path, or os.PathLike) to a .pdb file, load it.
    """

    if isinstance(pdb, mdj.Trajectory):
        return pdb

    if isinstance(pdb, (str, Path, os.PathLike)):
        pdb_path = str(pdb)
        if not Path(pdb_path).exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        return mdj.load_pdb(pdb_path)

    raise TypeError(f"Unsupported type for `pdb`: {type(pdb)}. "
                    "Expected mdtraj.Trajectory or path to a .pdb file.")


class DistanceFeatureGenerator(object): 
    def __init__(self, wepy_h5=None, pdb=None, lig_idxs=None, prot_idxs=None):

        self.wepy_h5 = wepy_h5
        self.pdb = _load_pdb(pdb)
        self.topology = pdb.topology
        self.lig_idxs = self.topology.select(lig_idxs)
        
        # in case receptor is a heterodimer
        if isinstance(prot_idxs, (list, tuple)):
            self.prot_sel = prot_idxs
        else:
            self.prot_sel = [prot_idxs]

        self.prot_idxs_list = [self.topology.select(sel) for sel in self.prot_sel]
        self.prot_idxs_combined = np.concatenate(self.prot_idxs_list)

    def _group_coordinates(self, frame_pos, box_vectors):
        # start from raw positions
        coords = frame_pos

        # If multiple receptor segments, group them first
        if len(self.prot_idxs_list) == 2:
            idxA, idxB = self.prot_idxs_list
            coords = group_pair(coords, box_vectors, idxA, idxB)

        # Now group ligand around that combined receptor
        grouped_coords = group_pair(coords, box_vectors, self.lig_idxs, self.prot_idxs_combined)

        return grouped_coords

    def compute_ligand_rmsd(self, fields, *args):

        " Compute the RMSD of a ligand after aligning on the starting pose of the receptor."

        pos = fields['positions']  
        bvs = np.diagonal(fields['box_vectors'], axis1=1, axis2=2)

        rmsds = []
        for i in range(len(pos)):
            # Group protein complexes if it's a heterodimer
            grouped_pos = self._group_coordinates(pos[i], bvs[i])
            centered_pos = center_around(grouped_pos, self.prot_idxs_combined)
            # Superimpose the protein-ligand complex onto the initial pose
            impose_pos, _, _ = superimpose(self.pdb.xyz[0], centered_pos, self.prot_idxs_combined)
            # calculate RMSD
            sup_rmsd = calc_rmsd(self.pdb.xyz[0], impose_pos, idxs=self.lig_idxs)
            rmsds.append(sup_rmsd)

        return np.array(rmsds)
    
    def compute_pairwise_dist(self, fields, *args):

        "Compute all pairwise distances between two groups of atoms for each frame."

        pairs_list = args[0]
        group1 = pairs_list[0]  
        group2 = pairs_list[1]  
        pos = fields['positions']
        bvs = np.diagonal(fields['box_vectors'], axis1=1, axis2=2)

        pair_dist = np.zeros((len(pos), group1.shape[0] * group2.shape[0]))
        for i in range(len(pos)):
            grouped_pos = self._group_coordinates(pos[i], bvs[i])

            counter = 0
            for g1_atom in group1:
                for g2_atom in group2:
                    pair_dist[i][counter] = np.sqrt(np.sum(np.square(grouped_pos[g1_atom] - grouped_pos[g2_atom])))
                    counter += 1

        return pair_dist

    def compute_pair_distances(self, fields, *args):

        "Compute distances between specific atom groups for each frame."

        pairs_list = [args[0]]
        pos = fields['positions']
        bvs = np.diagonal(fields['box_vectors'], axis1=1, axis2=2)

        pair_dist = np.zeros((len(pos), len(pairs_list)))
        for i in range(len(pos)):
            grouped_pos = self._group_coordinates(pos[i], bvs[i])

            for idx, pairs in enumerate(pairs_list):
                # print(f'Caulculating hbonds for pair: {idx}')
                pos_atom1 =  grouped_pos[pairs[0]]
                pos_atom2 =  grouped_pos[pairs[1]]

                pair_dist[i, idx] += LA.norm(pos_atom1 - pos_atom2)

        return pair_dist

    def minimum_distance(self, coordsA, coordsB):
        """
        Calculate the minimum distance between members of coordsA and coordsB.
        Uses a fast binary search algorithm with computational cost of the 
        order of N*log(M), where N=dim(A) and M=dim(B), instead of normal N*M cost.
        Will be useful for cases where we have large number of binding site residues.

        Parameters
        ----------

        coordsA : arraylike, shape (Natoms_A, 3)
            First set of coordinates.

        coordsB : arraylike, shape (Natoms_B, 3)
            Second set of coordinates.

        """

        # make sure the number of dimensions is 3
        assert (coordsA.shape[1] == 3) and (coordsB.shape[1] == 3), \
            "Minimum distance expecting arrays of shape (N, 3)"
        
        tree = KDTree(coordsA)

        return(tree.query(coordsB, dualtree=False, k=1)[0]).min()

    def compute_min_dist(self, fields, *args):
        group1 = args[0]
        group2 = args[1]

        pos = fields['positions']
        bvs = np.diagonal(fields['box_vectors'], axis1=1, axis2=2)

        min_dist = np.zeros((len(pos)))
        for i in range(len(pos)):
            grouped_pos = self._group_coordinates(pos[i], bvs[i])

            min_dist[i] = self.minimum_distance(grouped_pos[group1],
                                                grouped_pos[group2])

        return min_dist

    def call_compute_ligand_rmsd(self, out_file_name, out_folder_name, save_file=True):
        out_folder = Path(out_folder_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        save_file_name = osp.join(out_folder, out_file_name)

        if osp.exists(save_file_name):
            print(f'Warning! File {save_file_name} already exists, skipping to not re-write!')
            return None

        print('Calculating RMSDs',flush=True)
        t1 = time.time()

        with self.wepy_h5:
            ligand_rmsd = self.wepy_h5.compute_observable(self.compute_ligand_rmsd,
                                                        ['positions','box_vectors'],
                                                        (),
                                                        return_results=True
                                                        )

        if save_file:
            np.save(save_file_name, np.array(ligand_rmsd))
        
        t2 = time.time()
        print('Done calculating the RMSDs in',t2-t1,'seconds')

        return ligand_rmsd


    def call_compute_pairwise_distances(self, pairs_list, out_file_name, out_folder_name, save_file=True):   
        out_folder = Path(out_folder_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        save_file_name = osp.join(out_folder, out_file_name)

        if osp.exists(save_file_name):
            print(f'Warning! File {save_file_name} already exists, skipping to not re-write!')
            return None

        print(f'Calculating pairwise distances for {save_file_name}')
        t1 = time.time()

        with self.wepy_h5:
            pair_dist = self.wepy_h5.compute_observable(self.compute_pairwise_dist,
                                                            ['positions','box_vectors'],
                                                            (pairs_list),
                                                            return_results=True
                                                            )

        if save_file:
            np.save(save_file_name, np.array(pair_dist))

        t2 = time.time()
        print('Done calculating the pairwise distances in ',t2-t1,' seconds\n')

        return pair_dist

    def call_compute_pair_distances(self, pairs_list, out_file_name, out_folder_name, save_file=True):   
        out_folder = Path(out_folder_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        save_file_name = osp.join(out_folder, out_file_name)

        if osp.exists(save_file_name):
            print(f'Warning! File {save_file_name} already exists, skipping to not re-write!')
            return None

        print(f'Calculating  pair distances for {save_file_name}')
        t1 = time.time()

        with self.wepy_h5:
            pair_dist = self.wepy_h5.compute_observable(self.compute_pair_distances,
                                                            ['positions','box_vectors'],
                                                            (pairs_list),
                                                            return_results=True
                                                            )

        if save_file:
            np.save(save_file_name, np.array(pair_dist))

        t2 = time.time()
        print('Done calculating the pair distances in ',t2-t1,' seconds\n')

        return pair_dist

    def call_compute_min_dist(self, group1, group2, out_file_name, out_folder_name, save_file=True):   
        out_folder = Path(out_folder_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        save_file_name = osp.join(out_folder, out_file_name)

        if osp.exists(save_file_name):
            print(f'Warning! File {save_file_name} already exists, skipping to not re-write!')
            return None

        print(f'Calculating  minimum distances for {save_file_name}')
        t1 = time.time()

        with self.wepy_h5:
            min_dist = self.wepy_h5.compute_observable(self.compute_min_dist,
                                                        ['positions','box_vectors'],
                                                        (group1, group2),
                                                        return_results=True
                                                        )

        if save_file:
            np.save(save_file_name, np.array(min_dist))

        t2 = time.time()
        print('Done calculating minimum distances in ',t2-t1,' seconds\n')

        return min_dist
