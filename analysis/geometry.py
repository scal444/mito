import numpy as np


class mito_dims:
    ''' Helper structure for mito dimensions '''
    def __init__(self, l_cylinder, r_cylinder, r_junction, l_flat):
        self.l_cylinder = l_cylinder
        self.r_cylinder = r_cylinder
        self.r_junction = r_junction
        self.l_flat  = l_flat


def get_mito_center(coords, dims):
    '''
        For the z center, can just take the midpoint of the maximum and minimum coordinates, as the top and bottom
        plane z values are regular. For x/y, we have to potentially account for e.g. a cutoff of dummies along the x or
        y extreme that would throw off the COM by a tiny amount.

        Instead, we'll take the z com, and then center the x/y on the com of the cylindrical region. We don't need to
        worry about getting a little bit of the junction at the top and bottom of the cylinder - it's regular about the
        center just as the cylinder is.

        Parameters:
            coords - n_particles * 3 array
            dims   = mito_shape instance
        Returns the center as a np.array(3), doesn't make transformation
    '''
    z_com = (coords[:, 2].max() - coords[:, 2].min()) / 2.0
    xy_inrange = np.abs(coords[:, 2] - z_com) < dims.l_cylinder / 2
    xy_means = coords[xy_inrange, 0:2].mean(axis=0)
    return np.array((xy_means[0], xy_means[1], z_com))


def assign_to_mito_section(rho, z, mito_dims):
    ''' For one frame, generates an assignment_indices instance assigning inner and outer indices to the three
        regions of the mitochondria

        This is based on a specific geometry of mitochondria, from mito dims

        Parameters
            rho - radial paramater (1D - n_parts)
            z   - z dimension      (1D - n_parts)
            mito_dims - mito shape object
        Returns
            in_cylinder - list of indices
            in_flat     - list of indices
            in_junction - list of indices. union of in_cylinder, in_flat, in_junction is 0 to rho.size - 1

    '''
    in_cylinder = np.where(np.abs(z) < (mito_dims.l_cylinder / 2))[0]
    in_flat     = np.where(rho > (mito_dims.r_cylinder + mito_dims.r_junction))[0]
    in_junction = np.where((np.abs(z) >= (mito_dims.l_cylinder / 2)) &
                           (rho <= (mito_dims.r_cylinder + mito_dims.r_junction)))[0]
    return in_cylinder, in_flat, in_junction


def map_to_cyl_regime(z, mito_shape):
    ''' The mapping of the cylindrical regime is just 0 = center, l_cylinder  / 2 = edge'''
    return np.abs(z)


def map_to_flat_regime(rho, mito_shape):
    ''' rho of 0 is the edge of the junction - no outer edge '''
    return rho - mito_shape.r_cylinder - mito_shape.r_junction


def map_to_junction_regime(z, rho, mito_shape):
    ''' Angular description only to start - as a distance based one would require averaging the radii of all particles
       in the description. 0 degrees  = cylinder end, 90 degrees = flat end
    '''

    # first center coordinates relative to "circle" that describes the junction
    centered_z = np.abs(z) - (mito_shape.l_cylinder / 2)
    centered_rho = rho - mito_shape.r_cylinder - mito_shape.r_junction

    # coordinates are now in second quadrant of circle, subtract by 90 to get to convention
    # remember than arctan2 goes (y, x)
    return 180 - np.arctan2(centered_z, centered_rho) * 180 / np.pi


def map_to_unified_coordinate(z, rho, mito_shape):
    in_cyl, in_flat, in_junc = assign_to_mito_section(rho, z, mito_shape)

    junc_offset = mito_shape.l_cylinder / 2
    flat_offset = junc_offset + np.pi * mito_shape.r_junction / 2

    unified_coordinate = np.zeros(z.shape)
    unified_coordinate[in_cyl] = np.abs(z[in_cyl])
    unified_coordinate[in_junc] = map_to_junction_regime(z[in_junc], rho[in_junc], mito_shape) * mito_shape.r_junction / (180 / np.pi) + junc_offset  # noqa
    unified_coordinate[in_flat] = map_to_flat_regime(rho[in_flat], mito_shape) + flat_offset

    return unified_coordinate
