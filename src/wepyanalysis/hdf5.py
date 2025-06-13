from wepy.hdf5 import WepyHDF5
import os.path as osp

def link_wepy_runs(hdf5_path, input_files):
    with WepyHDF5(osp.join(hdf5_path, input_files[0]), mode='r+') as base:
        prev_new_idxs = list(base.run_idxs)

        for infile in input_files[1:]:
            # link in all continuations to the first run
            new_idxs = base.link_file_runs(osp.join(hdf5_path, infile))
            print(f"Imported {infile} to base index {new_idxs}")

            # if this isn’t the very first file, chain it to the previous
            if prev_new_idxs is not None:
                cont_run = new_idxs[0]
                base_run = prev_new_idxs[-1]
                base.add_continuation(continuation_run=cont_run,
                                        base_run=base_run)
                print(f"Chained run {cont_run} to run {base_run}")

            # idx for the next iteration
            prev_new_idxs = new_idxs
