"""
Generates a dummy variable-length image dataset in an HDF5 file
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2015, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import numpy
import tables
from pylearn2.utils.rng import make_np_rng

rng = make_np_rng(default_seed=123522)


if __name__ == "__main__":
    filters = tables.Filters(complib='blosc', complevel=5)
    h5file = tables.open_file('dummy.h5', mode='w',
                              title='Dummy variable length dataset',
                              filters=filters)
    group = h5file.create_group(h5file.root, 'Data', 'Data')
    atom = tables.UInt8Atom()
    X = h5file.create_vlarray(group, 'X', atom=atom, title='Data values',
                              expectedrows=500, filters=filters)
    y = h5file.create_carray(group, 'y', atom=atom, title='Data targets',
                             shape=(500, 1), filters=filters)
    s = h5file.create_carray(group, 's', atom=atom, title='Data shapes',
                             shape=(500, 3), filters=filters)

    shapes = rng.randint(low=10, high=101, size=(500, 2))
    for i, shape in enumerate(shapes):
        size = (shape[0], shape[1], 3)
        image = rng.uniform(low=0, high=1, size=size)
        target = rng.randint(low=0, high=2)

        X.append(image.flatten())
        y[i] = target
        s[i] = numpy.array(size)
        if i % 100 == 0:
            print i
            h5file.flush()
    h5file.flush()
