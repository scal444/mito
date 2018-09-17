import numpy as np
import mdtraj as md
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


# ----------------------------------------------------------------------------------------------------------------------
# mito coordinate loading
# ---------------------------------------------------------------------------------------------------------------------


class mito_coordinates:
    ''' Compiled coordinate description of mito system. Can be used for dummy or lipid data'''
    def __init__(self, theta, rho, z, unified):
        self.theta = theta
        self.rho = rho
        self.z = z
        self.unified = unified


class dummy_leaflet_data:
    def __init__(self, mito_coordinates, mito_shape, forces=None):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # n_frames * nparts * 3 array
        if forces:
            self.forces = forces


class lipid_leaflet_data:
    def __init__(self, mito_coordinates, lipid_indices, mito_shape):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # dict of indices
        self.lipid_indices = lipid_indices
        # instance of mito_shape
        self.mito_shape = mito_shape


def cart_2_mito(coords, unitcell_lengths, mito_center):
    '''
        Does a cartesian to polar transformation on trajectory data, based on a given center point. Accounts for periodic
        boundaries by calling periodic.calc_vectors

    '''
    mito_center_scaled = mito_center[np.newaxis, :].repeat(coords.shape[1], axis=0)[np.newaxis, :, :]
    mito_vecs   = periodic.calc_vectors(mito_center_scaled, coords, unitcell_lengths)
    return transformations.cart2pol(mito_vecs.squeeze())


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
    return dummy_forces[:, dummy_inner, :], dummy_forces[:, dummy_inner, dummy_outer]


def load_all_dummy_info(pdbpath, forcepath, indexpath, mito_shape, zo):
    inner_coords, outer_coords = load_and_split_dummy_pdb()



def load_mito_traj(trajpath, pdbpath, mito_shape, center):
    traj = md.load(trajpath, top=pdbpath)
    theta, rho, z = cart_2_mito(traj.xyz, traj.unitcell_lengths, mito_shape, center)
    unified_coord = geometry.map_to_unified_coordinate(z, rho, mito_shape)
    return mito_coordinates(theta, rho, z, unified_coord)


def process_dummy_leaflet_forces(unified_coord, forces, bins):
    force_avg_per_bead = np.sqrt((forces.mean(axis=0) ** 2).sum(axis=1))
    bin_assignments = np.digitize(unified_coord, bins)
    forces_by_bin = np.zeros(len(bins) - 1)
    forces_errors_by_bin   = np.zeros(len(bins) - 1)
    for bin_ind in range(1, len(bins)):
        force_ind = bin_ind - 1
        forces_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].mean()
        forces_errors_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].std() / np.sqrt(force_avg_per_bead[bin_assignments == bin_ind].size)
    return forces_by_bin, forces_errors_by_bin


def process_dummy_system(path, dummy_thickness, mito_shape, forcepath='/force.xvg', indexpath='/index.ndx',
                         coordpath='/dummy_only.pdb', firstframe=0):

    dummy_forces, dummy_pdb, top_dummy_ind, bot_dummy_ind = load_dummy_data(forcepath=path + forcepath,
                                                                            coordpath=path + coordpath,
                                                                            indexpath=path + indexpath)
    outer_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,                   mito_shape.r_cylinder + dummy_thickness,
                                          mito_shape.r_junction - dummy_thickness, mito_shape.l_flat)
    inner_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,                   mito_shape.r_cylinder - dummy_thickness,
                                          mito_shape.r_junction + dummy_thickness, mito_shape.l_flat)
    theta, rho, z = cart_2_mito(dummy_pdb, mito_shape)
    partitioned_dummy_indices = assign_dummy_particles_to_section(rho, z, top_dummy_ind, bot_dummy_ind, mito_shape)
    outer_unified_coord = geometry.map_to_unified_coordinate(z[bot_dummy_ind], rho[bot_dummy_ind], outer_mito_shape)
    inner_unified_coord = geometry.map_to_unified_coordinate(z[top_dummy_ind], rho[top_dummy_ind], inner_mito_shape)

    bin_spacing = 1
    outer_bins = np.arange(0, outer_unified_coord.max() + bin_spacing / 2, bin_spacing)
    inner_bins = np.arange(0, inner_unified_coord.max() + bin_spacing / 2, bin_spacing)
    outer_bin_centers = (outer_bins[1:] + outer_bins[:-1]) / 2
    inner_bin_centers = (inner_bins[1:] + inner_bins[:-1]) / 2

    inner_force_means, inner_force_errors = process_dummy_leaflet_forces(inner_unified_coord,
                                                                         dummy_forces[firstframe:, partitioned_dummy_indices.inner, :],
                                                                         inner_bins)
    outer_force_means, outer_force_errors = process_dummy_leaflet_forces(outer_unified_coord,
                                                                         dummy_forces[firstframe:, partitioned_dummy_indices.outer, :],
                                                                         outer_bins)

    outer_rho = [geometry.unified_to_rho(outer_mito_shape, i) for i in outer_bin_centers]
    inner_rho = [geometry.unified_to_rho(inner_mito_shape, i) for i in inner_bin_centers]
    outer_z = [geometry.unified_to_z(outer_mito_shape, i) for i in outer_bin_centers]
    inner_z = [geometry.unified_to_z(inner_mito_shape, i) for i in inner_bin_centers]
    return dummy_data(inner_rho, inner_z, inner_unified_coord, inner_force_means, outer_rho, outer_z, outer_unified_coord, outer_force_means )


if __name__ == '__main__':
    # project directory setup
    top_dir               = '/home/kevin/hdd/Projects/mito/'
    analysis_dir          = top_dir + 'analysis/'
    intermediate_data_dir = top_dir + 'data/'
    raw_data_dir          = top_dir + 'simulations/'

    firstframe = 1000
    # mito geometry info
    dummy_zo = 6.5 / 2
    mito_shape = geometry.mito_dims(30, 10, 10, 56)
    inner_mito_shape = geometry.dims(30, 10 - dummy_zo, 10 + dummy_zo, 56)
    outer_mito_shape = geometry.dims(30, 10 + dummy_zo, 10 - dummy_zo, 56)

    top_dummy_ind, bot_dummy_ind = load_dummy_indices(dummy_index_path)
    # get center, calculate vectors from center
    theta, rho, z = cart_2_mito(dummy_pdb, mito_shape)
    partitioned_dummy_indices = assign_dummy_particles_to_section(rho, z, top_dummy_ind, bot_dummy_ind, mito_shape)
    top_unified_coord = geometry.map_to_unified_coordinate(z[top_dummy_ind], rho[top_dummy_ind], top_mito_shape)
    bot_unified_coord = geometry.map_to_unified_coordinate(z[bot_dummy_ind], rho[bot_dummy_ind], bot_mito_shape)

    bin_spacing = 1
    top_bins = np.arange(0, top_unified_coord.max() + bin_spacing / 2, bin_spacing)
    bot_bins = np.arange(0, bot_unified_coord.max() + bin_spacing / 2, bin_spacing)
    top_bin_centers = (top_bins[1:] + top_bins[:-1]) / 2
    bot_bin_centers = (bot_bins[1:] + bot_bins[:-1]) / 2
    # double areas to account for up/down
    top_bin_areas = 2 * np.array([geometry.calc_unified_section_area(top_mito_shape, i, i + 1) for i in top_bins[:-1]])
    bot_bin_areas = 2 * np.array([geometry.calc_unified_section_area(bot_mito_shape, i, i + 1) for i in bot_bins[:-1]])
    top_bin_counts = np.bincount(np.digitize(top_unified_coord, top_bins))[1:]
    bot_bin_counts = np.bincount(np.digitize(bot_unified_coord, bot_bins))[1:]

    inner_force_means, inner_force_errors = process_dummy_leaflet_forces(top_unified_coord,
                                                                         dummy_forces[firstframe:, partitioned_dummy_indices.inner, :],
                                                                         top_bins)
    outer_force_means, outer_force_errors = process_dummy_leaflet_forces(bot_unified_coord,
                                                                         dummy_forces[firstframe:, partitioned_dummy_indices.outer, :],
                                                                         bot_bins)

    top_rho = [geometry.unified_to_rho(top_mito_shape, i) for i in top_bin_centers]
    bot_rho = [geometry.unified_to_rho(bot_mito_shape, i) for i in bot_bin_centers]
    top_z = [geometry.unified_to_z(top_mito_shape, i) for i in top_bin_centers]
    bot_z = [geometry.unified_to_z(bot_mito_shape, i) for i in bot_bin_centers]



    max_force = 0.0699768 * 4


    popc_path = '/home/kevin/hdd/Projects/mito/simulations/dummy_frozen/POPC_100'
    dope_path = '/home/kevin/hdd/Projects/mito/simulations/dummy_frozen/POPC80_DOPE20'
    tocl_path = '/home/kevin/hdd/Projects/mito/simulations/dummy_frozen/POPC80_TOCL20'

    mito_shape = geometry.mito_dims(30, 10, 10, 56)
    firstframe = 2000
    popc_data = process_dummy_system(popc_path, 6.5 / 2, mito_shape, firstframe=firstframe)
    tocl_data = process_dummy_system(tocl_path, 6.5 / 2, mito_shape, firstframe=firstframe)
    dope_data = process_dummy_system(dope_path, 6.5 / 2, mito_shape, firstframe=firstframe)

    plt.figure()
    plt.scatter(tocl_data.inner_rho, tocl_data.inner_z, c=tocl_data.inner_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.scatter(tocl_data.outer_rho, tocl_data.outer_z, c=tocl_data.outer_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.title('tocl')
    plt.xlabel("rho (nm)")
    plt.ylabel("z (nm)")
    cbar = plt.colorbar()
    cbar.set_label("pressure (kJ / (mol nm ^3))")

    plt.figure()
    plt.scatter(dope_data.inner_rho, dope_data.inner_z, c=dope_data.inner_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.scatter(dope_data.outer_rho, dope_data.outer_z, c=dope_data.outer_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.title('dope')
    plt.xlabel("rho (nm)")
    plt.ylabel("z (nm)")
    cbar = plt.colorbar()
    cbar.set_label("pressure (kJ / (mol nm ^3))")

    plt.figure()
    plt.scatter(popc_data.inner_rho, popc_data.inner_z, c=popc_data.inner_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.scatter(popc_data.outer_rho, popc_data.outer_z, c=popc_data.outer_force * 4, cmap='Reds', vmin=0, vmax=max_force)
    plt.title('popc')
    plt.xlabel("rho (nm)")
    plt.ylabel("z (nm)")
    cbar = plt.colorbar()
    cbar.set_label("pressure (kJ / (mol nm ^3))")
