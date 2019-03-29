import numpy as np
import mdtraj as md
import sys
sys.path.append('/home/kjb09011/python/')
import KB_python.file_io as file_io
import KB_python.coordinate_manipulation.periodic as periodic
import KB_python.coordinate_manipulation.transformations as transformations
import matplotlib.pyplot as plt
import pickle

import geometry

# --------------------------------------------------------------------------------------------------------------------
# pickle utilities
# --------------------------------------------------------------------------------------------------------------------


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------------------------------------------------
# index loading and processing
# ---------------------------------------------------------------------------------------------------------------------


def load_dummy_indices(file):
    ''' Assumes that bottom dummy section follows top dummy section immediately, and restarts numbering from 0'''
    indices = file_io.load_gromacs_index(file)
    inner_dummy_ind = np.array(indices['top_DUMY']) - indices['top_DUMY'][0]
    outer_dummy_ind = np.array(indices['bot_DUMY']) - indices['top_DUMY'][0]  # subtract from top! that's the 0 index
    return inner_dummy_ind, outer_dummy_ind


def load_leaflet_indices():
    pass


class mito_sectional_indices:
    def __init__(self, cylinder, junction,  flat):
        self.cylinder = cylinder
        self.junction = junction
        self.flat     = flat


def assign_dummy_particles_to_section(rho, z, mito_dims):
    in_cylinder, in_junction, in_flat = geometry.assign_to_mito_section(rho, z, mito_dims)
    return mito_sectional_indices(in_cylinder, in_junction, in_flat)


def load_lipid_indices(file, lipid_names):
    pass

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


dummy_leaflet_data = raw_dummy_leaflet_data


class processed_dummy_leaflet_data:
    def __init__(self, mito_coordinates, forces, force_errors):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # THIS IS TIME AVERAGED
        self.force = forces
        self.force_errors = force_errors


class lipid_leaflet_data:
    def __init__(self, mito_coordinates, lipid_indices, mito_shape):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # dict of indices
        self.lipid_indices = lipid_indices
        # instance of mito_shape
        self.mito_shape = mito_shape


# ----------------------------------------------------------------------------------------------------------------------
# general coordinate processing
# ---------------------------------------------------------------------------------------------------------------------


def cart_2_mito(coords, unitcell_lengths, mito_center):
    '''
        Does a cartesian to polar transformation on trajectory data, based on a given center point. Accounts for periodic
        boundaries by calling periodic.calc_vectors

    '''
    mito_center_scaled = mito_center[np.newaxis, :].repeat(coords.shape[1], axis=0)[np.newaxis, :, :]
    if coords.shape[0] > 1:
        mito_center_scaled = mito_center_scaled.repeat(coords.shape[0], axis=0)
    mito_vecs   = periodic.calc_vectors(mito_center_scaled, coords, unitcell_lengths)
    return transformations.cart2pol(mito_vecs.squeeze())


# ---------------------------------------------------------------------------------------------------------------------
# dummy loading functions
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
    theta, rho, z = cart_2_mito(dummy_pdb.xyz, dummy_pdb.unitcell_lengths, center)

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


# ---------------------------------------------------------------------------------------------------------------------
# coordinate analysis
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# main analysis
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # project directory setup
    top_dir               = '/home/kevin/hdd/Projects/mito/'
    analysis_dir          = top_dir + 'analysis/'
    intermediate_data_dir = top_dir + 'data/'
    raw_data_dir          = top_dir + 'simulations/'

    # mito geometry info
    dummy_zo = 6.5 / 2
    mito_shape = geometry.mito_dims(30, 10, 10, 56)
    inner_mito_shape = geometry.mito_dims(30, 10 - dummy_zo, 10 + dummy_zo, 56)
    outer_mito_shape = geometry.mito_dims(30, 10 + dummy_zo, 10 - dummy_zo, 56)

    # Shouldn't have to reload raw data every time - instead, load from pickle
    load_raw = False
    if load_raw:
        # POPC test case
        PC_path = raw_data_dir + 'dummy_frozen/POPC_100/'
        POPC_inner_dummy_data, POPC_outer_dummy_data = load_all_dummy_info(PC_path + 'dummy_only.pdb', PC_path + 'forces_further_reduced.xvg',
                                                                           PC_path + 'index.ndx', mito_shape, dummy_zo)
        pickle_save(POPC_inner_dummy_data, PC_path + "inner_dummy_data.pkl")
        pickle_save(POPC_outer_dummy_data, PC_path + "outer_dummy_data.pkl")

        # POPE test case
        PE_path = raw_data_dir + 'dummy_frozen/POPC80_DOPE20/'
        DOPE_inner_dummy_data, DOPE_outer_dummy_data = load_all_dummy_info(PE_path + 'dummy_only.pdb', PE_path + 'forces_further_reduced.xvg',
                                                                           PE_path + 'index.ndx', mito_shape, dummy_zo)
        pickle_save(DOPE_inner_dummy_data, PE_path + "inner_dummy_data.pkl")
        pickle_save(DOPE_outer_dummy_data, PE_path + "outer_dummy_data.pkl")

        # TOCL test case
        CL_path = raw_data_dir + 'dummy_frozen/POPC80_TOCL20/'
        TOCL_inner_dummy_data, TOCL_outer_dummy_data = load_all_dummy_info(CL_path + 'dummy_only.pdb', CL_path + 'forces_further_reduced.xvg',
                                                                           CL_path + 'index.ndx', mito_shape, dummy_zo)
        pickle_save(TOCL_inner_dummy_data, CL_path + "inner_dummy_data.pkl")
        pickle_save(TOCL_outer_dummy_data, CL_path + "outer_dummy_data.pkl")

        CL_ions_path = raw_data_dir + 'dummy_frozen/POPC80_TOCL20_ions/'
        TOCL_ions_inner_dummy_data, TOCL_ions_outer_dummy_data = load_all_dummy_info(CL_ions_path + 'dummy_only.pdb', CL_ions_path + 'forces_reduced.xvg',
                                                                                     CL_ions_path + 'index.ndx', mito_shape, dummy_zo)
        pickle_save(TOCL_ions_inner_dummy_data, CL_ions_path + "inner_dummy_data.pkl")
        pickle_save(TOCL_ions_outer_dummy_data, CL_ions_path + "outer_dummy_data.pkl")

    # analysis parameters
    firstframe = 100
    bin_width = 0.1

    POPC_inner_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC_100/inner_dummy_data.pkl')
    POPC_outer_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC_100/outer_dummy_data.pkl')
    POPC_inner_processed = process_dummy_system(POPC_inner_dummy_data, 1, firstframe=300)
    POPC_outer_processed = process_dummy_system(POPC_outer_dummy_data, 1, firstframe=300)

    TOCL_inner_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_TOCL20/inner_dummy_data.pkl')
    TOCL_outer_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_TOCL20/outer_dummy_data.pkl')
    TOCL_inner_processed = process_dummy_system(TOCL_inner_dummy_data, 1, firstframe=400)
    TOCL_outer_processed = process_dummy_system(TOCL_outer_dummy_data, 1, firstframe=400)

    DOPE_inner_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_DOPE20/inner_dummy_data.pkl')
    DOPE_outer_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_DOPE20/outer_dummy_data.pkl')
    DOPE_inner_processed = process_dummy_system(DOPE_inner_dummy_data, 1, firstframe=300)
    DOPE_outer_processed = process_dummy_system(DOPE_outer_dummy_data, 1, firstframe=300)

    TOCL_ions_inner_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_TOCL20_ions/inner_dummy_data.pkl')
    TOCL_ions_outer_dummy_data = pickle_load(raw_data_dir + 'dummy_frozen/POPC80_TOCL20_ions/outer_dummy_data.pkl')
    TOCL_ions_inner_processed = process_dummy_system(TOCL_ions_inner_dummy_data, 1, firstframe=600)
    TOCL_ions_outer_processed = process_dummy_system(TOCL_ions_outer_dummy_data, 1, firstframe=600)

    def plot_processed_data(inner_data, outer_data, cmax, cmap='Reds'):
        plt.figure()
        plt.scatter(inner_data.coordinates.rho, inner_data.coordinates.z, vmin=0, vmax=cmax, c=4 * inner_data.force, cmap=cmap)
        plt.scatter(outer_data.coordinates.rho, outer_data.coordinates.z, vmin=0, vmax=cmax, c=4 * outer_data.force, cmap=cmap)
        plt.xlabel("rho (nm)")
        plt.ylabel("z (nm)")
        plt.colorbar()
        plt.show()

    plot_processed_data(TOCL_ions_inner_processed, TOCL_ions_outer_processed, 0.35)
    plot_processed_data(TOCL_inner_processed, TOCL_outer_processed, 0.35)
    plot_processed_data(DOPE_inner_processed, DOPE_outer_processed, 0.35)
    plot_processed_data(POPC_inner_processed, POPC_outer_processed, 0.35)

    plt.figure()
    plt.plot(POPC_inner_processed.coordinates.unified, 4 * POPC_inner_processed.force, 'r-', label='100% POPC')
    plt.plot(DOPE_inner_processed.coordinates.unified, 4 * DOPE_inner_processed.force, 'r-', label='20% DOPE')
    plt.plot(TOCL_inner_processed.coordinates.unified, 4 * TOCL_inner_processed.force, 'r-', label='20% TOCL')
    plt.plot(TOCL_ions_inner_processed.coordinates.unified, 4 * TOCL_ions_inner_processed.force, 'r-', label='20% TOCL - ions')
    plt.xlabel("unified coordinate (nm)")
    plt.ylabel("Pressure (kJ/(mol nm^3)")
    plt.legend()
