import numpy as np
import unittest
import sys
sys.path.append('..')
import geometry  # noqa


class test_assign_to_mito_section(unittest.TestCase):

    def test_assign_to_mito_section(self):

        # load data
        mito_shape = geometry.mito_dims(30, 10, 10, 56)
        refprefix = 'ref_data/assign_to_mito_section/'
        rho = np.loadtxt(refprefix + 'rho.txt')
        z = np.loadtxt(refprefix + 'z.txt')

        cylinder, flat, junction = geometry.assign_to_mito_section(rho, z, mito_shape)

        # test size and that items have no intersect
        self.assertEqual(cylinder.size + junction.size + flat.size, rho.size)
        self.assertFalse(np.intersect1d(cylinder, flat))
        self.assertFalse(np.intersect1d(cylinder, junction))

        # test value sanity
        self.assertLess(        rho[cylinder].max(),       mito_shape.r_cylinder + mito_shape.r_junction)
        self.assertLessEqual(   np.abs(z[cylinder]).max(), mito_shape.l_cylinder / 2)
        self.assertGreaterEqual(np.abs(z[junction]).min(), mito_shape.l_cylinder / 2)
        self.assertLessEqual(   rho[junction].max(),       mito_shape.r_cylinder + mito_shape.r_junction)


class test_get_mito_center(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
