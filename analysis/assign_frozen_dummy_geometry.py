import numpy as np
import mdtraj as md
import sys
import os
sys.path.append('/home/kjb09011/python/')
import KB_python.file_io as file_io
from plotting import plot_dummy_forces_spatial
from utilities import pickle_load, pickle_save, string_id
import geometry


# ---------------------------------------------------------------------------------------------------------------------
# index loading and processing
# ---------------------------------------------------------------------------------------------------------------------


def load_dummy_indices(file):
    ''' Assumes that bottom dummy section follows top dummy section immediately, and restarts numbering from 0'''
    indices = file_io.load_gromacs_index(file)
    inner_dummy_ind = np.array(indices['top_DUMY']) - indices['top_DUMY'][0]
    outer_dummy_ind = np.array(indices['bot_DUMY']) - indices['top_DUMY'][0]  # subtract from top! that's the 0 index
    return inner_dummy_ind, outer_dummy_ind


# ---------------------------------------------------------------------------------------------------------------------
# load and process forces
# ---------------------------------------------------------------------------------------------------------------------


def load_raw_forces(path):
    return file_io.xvg_2_coords(file_io.load_large_text_file(path), 3)


# ---------------------------------------------------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------------------------------------------------

class mito_coordinates:
    ''' Compiled coordinate description of mito system. Can be used for dummy or lipid data'''
    def __init__(self, theta, rho, z, unified):
        self.theta = theta
        self.rho = rho
        self.z = z
        self.unified = unified


class raw_dummy_leaflet_data:
    def __init__(self, mito_coordinates, mito_shape, forces=None):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # n_frames * nparts * 3 array
        if forces is not None:
            self.forces = forces


class processed_dummy_leaflet_data:
    def __init__(self, mito_coordinates, forces, force_errors):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # THIS IS TIME AVERAGED
        self.force = forces
        self.force_errors = force_errors


# ---------------------------------------------------------------------------------------------------------------------
# functions for treating raw dummy data - should probably only call the load_and_serialize one
# ---------------------------------------------------------------------------------------------------------------------


def load_and_split_dummy_pdb(pdbpath, indexpath, mito_shape, zo):
    ''' Analyses should be split by leaflet, but it's easier with current setup to load the whole dummy system and then
        split into 2 leaflet data structures
    '''
    outer_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder + zo,
                                          mito_shape.r_junction - zo, mito_shape.l_flat)
    inner_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder - zo,
                                          mito_shape.r_junction + zo, mito_shape.l_flat)
    dummy_pdb = md.load(pdbpath)
    dummy_inner, dummy_outer = load_dummy_indices(indexpath)  # these are for dummy-only system, start at 0

    # center and transform whole system
    center = geometry.get_mito_center(dummy_pdb.xyz.squeeze(), mito_shape.l_cylinder)  # don't need inner/outer, this only uses l_cylinder
    theta, rho, z = geometry.cart_2_mito(dummy_pdb.xyz, dummy_pdb.unitcell_lengths, center)

    # unified coordinate is based on shape, so do separately
    inner_unified_coord = geometry.map_to_unified_coordinate(z[dummy_inner], rho[dummy_inner], inner_mito_shape)
    outer_unified_coord = geometry.map_to_unified_coordinate(z[dummy_outer], rho[dummy_outer], outer_mito_shape)

    inner_mito_coordinates =  mito_coordinates(theta.squeeze()[dummy_inner], rho.squeeze()[dummy_inner], z.squeeze()[dummy_inner], inner_unified_coord)
    outer_mito_coordinates =  mito_coordinates(theta.squeeze()[dummy_outer], rho.squeeze()[dummy_outer], z.squeeze()[dummy_outer], outer_unified_coord)
    return inner_mito_coordinates, outer_mito_coordinates


def load_and_split_dummy_forces(forcepath, indexpath):
    dummy_forces = load_raw_forces(forcepath)
    dummy_inner, dummy_outer = load_dummy_indices(indexpath)
    return dummy_forces[:, dummy_inner, :], dummy_forces[:, dummy_outer, :]


def load_all_dummy_info(pdbpath, forcepath, indexpath, mito_shape, zo):
    inner_coords, outer_coords = load_and_split_dummy_pdb(pdbpath, indexpath, mito_shape, zo)
    inner_forces, outer_forces = load_and_split_dummy_forces(forcepath, indexpath)

    outer_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder + zo,
                                          mito_shape.r_junction - zo, mito_shape.l_flat)
    inner_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder - zo,
                                          mito_shape.r_junction + zo, mito_shape.l_flat)

    return raw_dummy_leaflet_data(inner_coords, inner_mito_shape, forces=inner_forces), raw_dummy_leaflet_data(outer_coords, outer_mito_shape, forces=outer_forces)


def load_and_serialize_raw_data(data_dir, save_dir, mito_geometry, dummy_zo, lipid,
                                pdb="dummy_only.pdb", index="index.ndx", forces="forces.xvg"):
    inner_dummy_data, outer_dummy_data = load_all_dummy_info(os.path.join(data_dir, pdb), os.path.join(data_dir, forces),
                                                             os.path.join(data_dir, index), mito_geometry, dummy_zo)

    save_str = "raw_dummy_{}".format(string_id(lipid, mito_geometry, dummy_zo))
    pickle_save(inner_dummy_data, os.path.join(save_dir, save_str + "_inner.pkl"))
    pickle_save(outer_dummy_data, os.path.join(save_dir, save_str + "_outer.pkl"))

# ---------------------------------------------------------------------------------------------------------------------
# force analysis
# ---------------------------------------------------------------------------------------------------------------------


def process_dummy_leaflet_forces(raw_dummy_data, bins, firstframe, lastframe):

    force_avg_per_bead = np.sqrt((raw_dummy_data.forces[firstframe:lastframe, :, :].mean(axis=0) ** 2).sum(axis=1))
    bin_assignments = np.digitize(raw_dummy_data.coordinates.unified, bins)
    forces_by_bin = np.zeros(len(bins) - 1)
    forces_errors_by_bin   = np.zeros(len(bins) - 1)
    for bin_ind in range(1, len(bins)):
        force_ind = bin_ind - 1
        forces_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].mean()
        forces_errors_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].std() / np.sqrt(force_avg_per_bead[bin_assignments == bin_ind].size)
    return forces_by_bin, forces_errors_by_bin


def process_dummy_system(raw_dummy_data, bin_spacing, firstframe=0, lastframe=None):
    if not lastframe:
        lastframe = raw_dummy_data.forces.shape[0]

    bins = np.arange(0, raw_dummy_data.coordinates.unified.max(), bin_spacing)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    force_means, force_errors = process_dummy_leaflet_forces(raw_dummy_data, bins, firstframe, lastframe=lastframe)
    z = geometry.unified_to_z(raw_dummy_data.mito_shape, bin_centers)
    rho = geometry.unified_to_rho(raw_dummy_data.mito_shape, bin_centers)
    bin_coords = mito_coordinates(0, rho, z, bin_centers)

    return processed_dummy_leaflet_data(bin_coords, force_means, force_errors)


def load_and_process_serialized_data(data_dir, out_dir, geometry, lipid, dummy_zo, bin_spacing, first_frame):
    id = string_id(lipid, geometry, dummy_zo)
    load_str = "raw_dummy_{}".format(id)
    save_str = "processed_dummy_{}".format(id)
    inner_dummy_data = pickle_load(os.path.join(data_dir, load_str + "_inner.pkl"))
    outer_dummy_data = pickle_load(os.path.join(data_dir, load_str + "_outer.pkl"))
    inner_processed = process_dummy_system(inner_dummy_data, bin_spacing, firstframe=first_frame)
    outer_processed = process_dummy_system(outer_dummy_data, bin_spacing, firstframe=first_frame)
    pickle_save(inner_processed, os.path.join(out_dir, save_str + "_inner.pkl"))
    pickle_save(outer_processed, os.path.join(out_dir, save_str + "_outer.pkl"))


def load_processed_data(data_dir, geometry, lipid, zo):
    id = string_id(lipid, geometry, zo)
    return (pickle_load(os.path.join(data_dir, "processed_dummy_{}_inner.pkl".format(id))),
            pickle_load(os.path.join(data_dir, "processed_dummy_{}_outer.pkl".format(id))))

# ---------------------------------------------------------------------------------------------------------------------
# main analysis
# -----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # project directory setup
    top_dir               = '/home/kevin/hdd/Projects/mito/'
    analysis_dir          = top_dir + 'analysis/'
    raw_data_dir          = top_dir + 'simulations/'

    # mito geometry info
    dummy_zo = 6.5 / 2
    mito_shape = geometry.mito_dims(30, 10, 10, 56)

    # step 1 - load (lots of) raw data
    load_raw = False
    if load_raw:
        save_dir = analysis_dir + "intermediate_data"
        load_and_serialize_raw_data(raw_data_dir + 'dummy_frozen/POPC_100', save_dir, mito_shape, dummy_zo, "POPC",
                                    forces="forces_further_reduced.xvg")
        load_and_serialize_raw_data(raw_data_dir + 'dummy_frozen/POPC80_DOPE20', save_dir, mito_shape, dummy_zo,
                                    "DOPE", forces="forces_further_reduced.xvg")
        load_and_serialize_raw_data(raw_data_dir + 'dummy_frozen/POPC80_TOCL20', save_dir, mito_shape, dummy_zo,
                                    "TOCL", forces="forces_further_reduced.xvg")

    # step 2 - load serialized data, process, save processed data
    load_serialized = False
    if load_serialized:
        # analysis parameters
        firstframe = 100
        bin_width = 1
        read_dir = analysis_dir + "intermediate_data"
        write_dir = analysis_dir + "processed_data"
        load_and_process_serialized_data(read_dir, write_dir, mito_shape, "POPC", dummy_zo, bin_width, firstframe)
        load_and_process_serialized_data(read_dir, write_dir, mito_shape, "DOPE", dummy_zo, bin_width, firstframe)
        load_and_process_serialized_data(read_dir, write_dir, mito_shape, "TOCL", dummy_zo, bin_width, firstframe)

    # step 3 visualize results, profit
    lipids = ("POPC", "DOPE", "TOCL")
    read_dir = analysis_dir + "processed_data"
    max_pressure = 0.35
    geometries = (geometry.mito_dims(30, 10, 10, 56), )
    for lipid in lipids:
        for geo in geometries:
            inner_data, outer_data = load_processed_data(read_dir, geo, lipid, dummy_zo)
            plot_dummy_forces_spatial(inner_data, outer_data, max_pressure, title=string_id(lipid, geo, dummy_zo))
