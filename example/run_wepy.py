import sys
import os
import os.path as osp
import pickle as pkl
import logging
from pathlib import Path
import time

import mdtraj as mdj
import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from wepy.reporter.reporter import Reporter
from wepy.hdf5 import WepyHDF5
from wepy.sim_manager import Manager
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.runners.openmm import OpenMMGPUWalkerTaskProcess, OpenMMRunner, OpenMMWalker, OpenMMState, gen_sim_state
from wepy.boundary_conditions.receptor import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.openmm import OpenMMRunnerDashboardSection
from wepy.reporter.revo.dashboard import REVODashboardSection
from wepy.reporter.receptor.dashboard import UnbindingBCDashboardSection
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.reporter.reporter import Reporter

class WalkersPickleReporter(Reporter):
     def __init__(self, save_dir='./', freq=100, num_backups=2):
         # the directory to save the pickles in                                                                                                         
         self.save_dir = save_dir
         # the frequency of cycles to backup the walkers as a pickle                                                                                    
         self.backup_freq = freq
         # the number of sets of walker pickles to keep, this will keep                                                                                 
         # the last `num_backups`                                                                                                                       
         self.num_backups = num_backups

     def init(self, *args, **kwargs):
         # make sure the save_dir exists                                                                                                                
         if not osp.exists(self.save_dir):
             os.makedirs(self.save_dir)
         # delete backup pickles in the save_dir if they exist                                                                                          
         else:
             for pkl_fname in os.listdir(self.save_dir):
                 os.remove(osp.join(self.save_dir, pkl_fname))
     def report(self, cycle_idx=None, new_walkers=None,
                **kwargs):
         # ignore all args and kwargs                                                                                                                   
         # total number of cycles completed                                                                                                             
         n_cycles = cycle_idx + 1
         # if the cycle is on the frequency backup walkers to a pickle                                                                                  
         if n_cycles % self.backup_freq == 0:
             pkl_name = "walkers_cycle_{}.pkl".format(cycle_idx)
             pkl_path = osp.join(self.save_dir, pkl_name)
             with open(pkl_path, 'wb') as wf:
                 pkl.dump(new_walkers, wf)
             # remove old pickles if we have more than the `num_backups`                                                                                
             if (cycle_idx // self.backup_freq) >= self.num_backups:
                 old_idx = cycle_idx - self.num_backups * self.backup_freq
                 old_pkl_fname = "walkers_cycle_{}.pkl".format(old_idx)
                 os.remove(osp.join(self.save_dir, old_pkl_fname))


## SYSTEM SETUP ----
# change the paths
base_path = Path('.')
out_folder = Path('.')
out_folder_pkl = osp.join('./','pkl')

Path(out_folder).mkdir(parents=True, exist_ok=True)
Path(out_folder_pkl).mkdir(parents=True, exist_ok=True)

openmm_path = base_path 
rst_path = base_path 

pdb_path = osp.join(openmm_path,'step3_input.pdb')
hdf5_out_path = osp.join(out_folder,'wepy.results0.h5')
dashboard_path = osp.join(out_folder,'wepy.dashall0.org')

hdf5_save_fields = ('positions','box_vectors')

with open(osp.join(openmm_path,'system.pkl'),'rb') as f:
    system = pkl.load(f)

with open(osp.join(openmm_path,'topology.pkl'),'rb') as f:
    omm_top = pkl.load(f)

# make an integrator object that is constant temperature
integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                    1/unit.picosecond,
                                    0.002*unit.picoseconds)

# Read .rst file
with open(rst_path, 'r') as f:
    simtk_state = omm.XmlSerializer.deserialize(f.read())

bv = simtk_state.getPeriodicBoxVectors()
system.setDefaultPeriodicBoxVectors(bv[0],bv[1],bv[2])

new_simtk_state = gen_sim_state(simtk_state.getPositions(),
                                system,
                                integrator)

omm_state = OpenMMState(new_simtk_state)
    
# get a json topology object
pdb = mdj.load_pdb(pdb_path)
json_top = mdtraj_to_json_topology(pdb.top)

lig_idxs = pdb.top.select('segname PROC')
protein_idxs = pdb.top.select('segname PROA or segname PROB')
protein_lig_idxs = pdb.top.select('segname PROA or segname PROB or segname PROC')
binding_selection_idxs =  mdj.compute_neighbors(pdb, 0.5, lig_idxs, haystack_indices=protein_idxs, periodic=True)[0]

## END SYSTEM SETUP ----

# set up the OpenMMRunner with your system
runner = OpenMMRunner(system, omm_top, integrator, platform='CUDA')

# set up parameters for running the simulation
num_walkers = 48
# initial weights
init_weight = 1.0 / num_walkers

# a list of the initial walkers
init_walkers = [OpenMMWalker(omm_state, init_weight) for i in range(num_walkers)]

unb_distance = UnbindingDistance(lig_idxs,
                                    binding_selection_idxs,
                                    omm_state)

# set up the REVO Resampler with the parameters
resampler = REVOResampler(merge_dist=0.25,
                            char_dist=0.1,
                            distance=unb_distance,
                            init_state=omm_state,
                            weights=True,
                            merge_alg="pairs",
                            pmax=0.1,
                            dist_exponent=4)

ubc = UnbindingBC(cutoff_distance=1.0,
                    initial_state=omm_state,
                    topology=json_top,
                    ligand_idxs=lig_idxs,
                    receptor_idxs=protein_idxs)

# instantiate a reporter for HDF5
reporter = WepyHDF5Reporter(save_fields=hdf5_save_fields,
                            file_path=hdf5_out_path,
                            resampler=resampler,
                            boundary_conditions=ubc,
                            topology=json_top)

# create a work mapper for NVIDIA GPUs for a GPU cluster
mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                    num_workers=4, 
                    platform='CUDA',
                    device_ids=[0,1,2,3]) #change this according to number of workers

# make REVO dashboard reporter
openmm_dashboard_sec = OpenMMRunnerDashboardSection(runner)
unb_bc_dashboard_sec = UnbindingBCDashboardSection(ubc)
revo_dashboard_sec = REVODashboardSection(resampler)
dashboard_reporter = DashboardReporter(file_path = dashboard_path,
                                        resampler_dash = revo_dashboard_sec,
                                        runner_dash = openmm_dashboard_sec,
                                        bc_dash = unb_bc_dashboard_sec)

# make walker pkl reporter
pkl_reporter = WalkersPickleReporter(save_dir = out_folder_pkl,
                                        freq = 1,
                                        num_backups = 2)

# Instantiate a simulation manager
sim_manager = Manager(init_walkers,
                        runner=runner,
                        resampler=resampler,
                        boundary_conditions=ubc,
                        work_mapper=mapper,
                        reporters=[reporter, reporter_nosolv, pkl_reporter, dashboard_reporter])

#------------------------------
# Run the simulation
#------------------------------

n_steps = 10000  
n_cycles = 250 

# run a simulation with the manager for n_steps cycles of length 1000 each
steps = [ n_steps for i in range(n_cycles)]
print("Running simulation")

begin = time.time()
sim_manager.run_simulation(n_cycles,
                            steps)

end = time.time()

print("Completed",n_cycles,"in",end-begin,"seconds")

# your data should be in the hdf5_out_path

