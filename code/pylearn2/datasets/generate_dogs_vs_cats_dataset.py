"""
Puts the Dogs vs. Cats dataset in an HDF5 file.
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import numpy
import tables
from os import listdir
from os.path import isfile, join
from scipy import misc
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng

rng = make_np_rng(default_seed=123522)


if __name__ == "__main__":
    base_dir = serial.preprocess(join('${PYLEARN2_DATA_PATH}', 'dogs_vs_cats'))
    files = [f for f in listdir(join(base_dir, 'train'))
             if isfile(join(base_dir, 'train', f))]

    filters = tables.Filters(complib='blosc', complevel=5)
    h5file = tables.open_file(join(base_dir, 'train.h5'), mode='w',
                              title='Dogs vs. Cats - Training set',
                              filters=filters)
    group = h5file.create_group(h5file.root, 'Data', 'Data')
    atom_8 = tables.UInt8Atom()
    atom_32 = tables.UInt32Atom()
    X = h5file.create_vlarray(group, 'X', atom=atom_8, title='Data values',
                              expectedrows=25000, filters=filters)
    y = h5file.create_carray(group, 'y', atom=atom_8, title='Data targets',
                             shape=(25000, 1), filters=filters)
    s = h5file.create_carray(group, 's', atom=atom_32, title='Data shapes',
                             shape=(25000, 3), filters=filters)

    # Shuffle examples around
    rng.shuffle(files)
    for i, f in enumerate(files):
        image = misc.imread(join(base_dir, 'train', f))
        X.append(image.flatten())
        target = 0 if 'cat' in f else 1
        y[i] = target
        s[i] = image.shape
        if i % 100 == 0:
            print i
            h5file.flush()
    h5file.flush()
