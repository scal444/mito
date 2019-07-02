import geometry
from  assign_frozen_dummy_geometry import mito_coordinates, cart_2_mito
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import pickle


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class lipid_leaflet_data:
    def __init__(self, mito_coordinates, lipid_indices, mito_shape):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # dict of indices
        self.lipid_indices = lipid_indices
        # instance of mito_shape
        self.mito_shape = mito_shape


class lipid_indices:
    def __init__(self, pdbpath):
        pdb = md.load(pdbpath)
        self.POPC = pdb.topology.select("resname POPC")
        self.DOPE = pdb.topology.select("resname DOPE")
        self.TOCL = pdb.topology.select("resname TOCL")
# ---------------------------------------------------------------------------------------------------------------------
# lipid loading functions
# ---------------------------------------------------------------------------------------------------------------------


def load_mito_traj(trajpath, pdbpath, mito_shape, center, stride=1):
    traj = md.load(trajpath, top=pdbpath, stride=stride)
    theta, rho, z = cart_2_mito(traj.xyz, traj.unitcell_lengths, center)
    unified_coord = geometry.map_to_unified_coordinate(z, rho, mito_shape)
    return mito_coordinates(theta, rho, z, unified_coord)


def POPC_fraction(digitized_lipids, lip_indices):
    n_frames = digitized_lipids.shape[0]
    n_bins = 1 + digitized_lipids.max() - digitized_lipids.min()
    PC_fract = np.zeros((n_frames, n_bins))

    for i in range(digitized_lipids.min(), digitized_lipids.max() + 1):
        digitized_mask = (digitized_lipids == i)
        PC_fract[:, i - 1] = digitized_mask[:, lip_indices.POPC].sum(axis=1) / digitized_mask.sum(axis=1)
    return PC_fract


def gen_distributed_bins(data, points_per_bin):
    bins = [0]
    sorted_data  = np.sort(data)
    # note that we don't bother getting the max value - when we digitize, anything right
    # of the maximum is just another data point
    return bins + [sorted_data[i] for i in range(points_per_bin, sorted_data.size, points_per_bin )]


def reduce_1st_dimension(data, averagerate):
    newdims = [*data.shape]
    newdims[0] = int(np.floor(float(data.shape[0]) / averagerate))
    reduced_array = np.zeros(newdims)
    for i in range(0, newdims[0]):
        reduced_array[i, :] = data[i * averagerate : (i + 1) * averagerate, :].mean(axis=0)
    return reduced_array


def reduce_2nd_dimension(data, averagerate):
        newdims = [*data.shape]
        newdims[1] = int(np.floor(float(data.shape[0]) / averagerate))
        reduced_array = np.zeros(newdims)
        for i in range(0, newdims[1]):
            reduced_array[:, i] = data[:, i * averagerate : (i + 1) * averagerate].mean(axis=1)
        return reduced_array


def plot_lipid_occupancy_heatmap(data, mindelta=-5, maxdelta=5, cmap='seismic', xlabels=None, ylabels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=mindelta, vmax=maxdelta, cmap=cmap, origin='lower')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    if xlabels is not None:
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticklabels(ylabels)
    ax.figure.colorbar(im, ax=ax, label="% change")
    plt.show()


def plot_lipid_occupancy_spatial(inner_data, inner_coords, outer_data, outer_coords, mindelta=-5, maxdelta=5, cmap='seismic'):
    plt.figure()
    plt.scatter(inner_coords.rho, inner_coords.z, c=inner_data, vmin=mindelta, vmax=maxdelta, cmap=cmap)
    plt.scatter(outer_coords.rho, outer_coords.z, c=outer_data, vmin=mindelta, vmax=maxdelta, cmap=cmap)
    plt.xlabel("Rho (nm)")
    plt.ylabel("z (nm)")
    plt.colorbar(label="% change")
    plt.show()


def load_leaflet_spatial_data(centerpath, datapath, prefix, mito_shape):
    center_coords = md.load(centerpath)
    mito_center = geometry.get_mito_center(center_coords.xyz.squeeze(), mito_shape.l_cylinder)
    coords = load_mito_traj(datapath + prefix + '.xtc', datapath + prefix + '.pdb', mito_shape, mito_center)
    indices = lipid_indices(datapath + prefix + '.pdb')
    return coords, indices


def digitize_leaflet_data(coordinates, nlips_per_bin):
    bins = np.array(gen_distributed_bins(coordinates.unified[0, :], nlips_per_bin))
    digitized_coords = np.digitize(coordinates.unified, bins)
    digitized_coords[digitized_coords == digitized_coords.max()] = digitized_coords.max() - 1

    # gather coordinate info on new bin centers
    nbins = 1 + digitized_coords.max() - digitized_coords.min()
    min_bin = digitized_coords.min()
    bin_unified = np.zeros(nbins)
    bin_z = np.zeros(nbins)
    bin_rho = np.zeros(nbins)

    for i in range(nbins):
        bin_unified[i] = coordinates.unified[digitized_coords == (i + min_bin)].mean()
        bin_z[i]       = np.abs(coordinates.z[digitized_coords == (i + min_bin)]).mean()
        bin_rho[i]     = coordinates.rho[digitized_coords == (i + min_bin)].mean()

    return digitized_coords, mito_coordinates([], bin_rho, bin_z, bin_unified)


def process_leaflet_spatial_data(coordinates, indices, nlips_per_bin):
    digitized_coords, bin_coordinates = digitize_leaflet_data(coordinates, nlips_per_bin)
    PC_fraction = POPC_fraction(digitized_coords, indices)
    return 1 - PC_fraction, bin_coordinates


def load_and_process_spatial_data(centerpath, datapath, prefix, mito_shape, n_lips_per_bin, dt=0, outdir='./', save=True):
    coords, indices = load_leaflet_spatial_data(centerpath, datapath, prefix, mito_shape)
    minor_component_fraction, bin_coordinates = process_leaflet_spatial_data(coords, indices, n_lips_per_bin)
    if dt > 0:
        minor_component_fraction = reduce_1st_dimension(minor_component_fraction, dt)
    if save:
        pickle_save(bin_coordinates, outdir + prefix + '_bin_centers.pkl')
        np.savetxt(outdir + prefix + '_minor_component_fraction.txt', minor_component_fraction)
    return minor_component_fraction, bin_coordinates


topdir  = '/home/kevin/hdd/Projects/mito/simulations/dummy_frozen/POPC80_DOPE20/python/'
mito_shape = geometry.mito_dims(30, 10, 10, 56)
inner_component = np.loadtxt(topdir + 'inner_hg_minor_component_fraction.txt')
inner_coords = pickle_load(topdir + 'inner_hg_bin_centers.pkl')
outer_component = np.loadtxt(topdir + 'outer_hg_minor_component_fraction.txt')
outer_coords = pickle_load(topdir + 'outer_hg_bin_centers.pkl')
