import numpy as np
import mdtraj as md
import KB_python.file_io as file_io
import KB_python.coordinate_manipulation.periodic as periodic
import KB_python.coordinate_manipulation.transformations as transformations

import geometry


class dummy_sectional_indices:
    def __init__(self, inner_cylinder, outer_cylinder, inner_junction, outer_junction, inner_flat, outer_flat):
        self.inner_cylinder = inner_cylinder
        self.outer_cylinder = outer_cylinder
        self.inner_junction = inner_junction
        self.outer_junction = outer_junction
        self.inner_flat     = inner_flat
        self.outer_flat     = outer_flat


def assign_dummy_particles_to_section(rho, z, top_ind, bot_ind, mito_dims):
    in_cylinder, in_flat, in_junction = geometry.assign_to_mito_section(rho, z, mito_dims)
    return dummy_sectional_indices(np.intersect1d(in_cylinder, top_ind), np.intersect1d(in_cylinder, bot_ind),
                                   np.intersect1d(in_junction, top_ind), np.intersect1d(in_junction, bot_ind),
                                   np.intersect1d(in_flat,     top_ind), np.intersect1d(in_flat,     bot_ind))


def load_dummy_indices(file):
    indices = file_io.load_gromacs_index('/home/kevin/hdd/Projects/mito/simulations/dummy/POPC_100/index.ndx')
    top_dummy_ind = np.array(indices['top_DUMY']) - indices['top_DUMY'][0]
    bot_dummy_ind = np.array(indices['bot_DUMY']) - indices['top_DUMY'][0]  # subtract from top! that's the 0 index
    return top_dummy_ind, bot_dummy_ind


if __name__ == '__main__':
    # project directory setup
    top_dir               = '/home/kevin/hdd/Projects/mito/'
    analysis_dir          = top_dir + 'analysis/'
    intermediate_data_dir = top_dir + 'data/'
    raw_data_dir          = top_dir + 'simulations/'

    # mito geometry info
    mito_shape = geometry.mito_dims(30, 10, 10, 56)

    # example data setup
    dummy_forces_path = raw_data_dir + 'frozen_dummy/dummy_force.xvg'
    dummy_coord_path  = raw_data_dir + 'dummy/POPC_100/dummy_only.pdb'
    dummy_index_path  = raw_data_dir + 'dummy/POPC_100/index.ndx'

    # load data
    dummy_forces = file_io.load_xvg(dummy_forces_path)
    dummy_pdb   = md.load(dummy_coord_path)
    dummy_coords = dummy_pdb.xyz.squeeze()
    top_dummy_ind, bot_dummy_ind = load_dummy_indices(dummy_index_path)

    # get center, calculate vectors from center
    mito_center = geometry.get_mito_center(dummy_pdb.xyz.squeeze(), mito_shape)
    mito_center_scaled = mito_center[np.newaxis, :].repeat(dummy_coords.shape[0], axis=0)[np.newaxis, :, :]
    mito_vecs   = periodic.calc_vectors(mito_center_scaled, dummy_coords[np.newaxis, :, :], dummy_pdb.unitcell_lengths)
    theta, rho, z = transformations.cart2pol(mito_vecs.squeeze())

    partitioned_dummy_indices = assign_dummy_particles_to_section(rho, z, top_dummy_ind, bot_dummy_ind, mito_shape)
    outer_cyl_coord = geometry.map_to_cyl_regime(z[partitioned_dummy_indices.outer_cylinder], mito_shape)
    outer_junc_coord = geometry.map_to_junction_regime(z[partitioned_dummy_indices.outer_junction],
                                                       rho[partitioned_dummy_indices.outer_junction], mito_shape)
    outer_flat_coord = geometry.map_to_flat_regime(rho[partitioned_dummy_indices.outer_flat], mito_shape)

    outer_cyl_force  = np.sqrt((dummy_forces[:, partitioned_dummy_indices.outer_cylinder, :] ** 2).sum(axis=2)).mean(axis=0)
    outer_junc_force = np.sqrt((dummy_forces[:, partitioned_dummy_indices.outer_junction, :] ** 2).sum(axis=2)).mean(axis=0)
    outer_flat_force = np.sqrt((dummy_forces[:, partitioned_dummy_indices.outer_flat, :] ** 2).sum(axis=2)).mean(axis=0)
