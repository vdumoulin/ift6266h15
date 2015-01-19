"""
A dataset containing images of variable sizes
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import numpy
import tables
from scipy import misc
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.iteration import SubsetIterator, resolve_iterator_class
from pylearn2.space import VectorSpace, IndexSpace, Conv2DSpace, CompositeSpace
from pylearn2.utils import safe_izip, wraps
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng
from pylearn2.datasets.dataset import Dataset


class VariableImageDataset(Dataset):
    """
    An HDF5 dataset which contains images of variable sizes which are
    preprocessed on-the-fly

    Parameters
    ----------
    path : str
        Path to the HDF5 file storing the dataset
    data_node : str
        Name of the data node in the file
    transformer : BaseTransformer
        Object that applies on-the-fly preprocessing to examples in the dataset
    X_str : str
        Name of the array containing flattened features
    s_str : str
        Name of the array containing image shapes
    y_str : str, optional
        Name of the array containing targets. Defaults to 'None', meaning no
        targets are available.
    y_labels : int
        Number of classes. Defaults to 'None', meaning targets are not
        categorical data.
    axes : tuple, optional
        Ordering of the axes. Must be a ('b', 0, 1, 'c') permutation. Defaults
        to ('b', 0, 1, 'c').
    rng : int or rng, optional
        RNG or seed for an RNG. Defaults to some default seed value.
    """
    _default_seed = 2015 + 1 + 17

    def __init__(self, path, data_node, transformer, X_str, s_str, y_str=None,
                 y_labels=None, axes=('b', 0, 1, 'c'), rng=_default_seed):
        path = preprocess(path)
        self.h5file = tables.openFile(path, mode="r")
        node = self.h5file.getNode('/', data_node)

        self.rng = make_np_rng(rng, which_method="random_integers")

        self.X = getattr(node, X_str)
        # Make sure images have values in [0, 1]. This is needed for
        # self.adjust_for_viewer, amongst other things.
        if not numpy.all(x >= 0 and x <= 1 for x in self.X.iterrows()):
            raise ValueError("features must be normalized between 0 and 1")
        self.num_examples = self.X.shape[0]
        self.s = getattr(node, s_str)
        self.y = getattr(node, y_str) if y_str is not None else None

        self.y_labels = y_labels
        self._check_labels()

        self.transformer = transformer

        X_source = 'features'
        shape = self.transformer.get_shape()
        channels = self.s[0][-1]
        X_space = Conv2DSpace(shape=shape, num_channels=channels, axes=axes)

        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            if self.y_labels is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            else:
                y_space = VectorSpace(dim=dim)
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)
        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class(
            'batchwise_shuffled_sequential')
        self._iter_data_specs = self.data_specs

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def _check_labels(self):
        """
        Sanity check for y_labels
        """
        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            assert numpy.all(l < self.y_labels for l in self.y.iterrows())

    def get(self, source, indexes):
        """
        Returns required examples for the required data sources, e.g. the first
        ten features and targets pairs or the last five targets

        Parameters
        ----------
        source : tuple of str
            Tuple of source names
        indexes : slice
            Examples to fetch
        """
        self._validate_source(source)
        rval = []
        for so in source:
            if so == 'features':
                images = self.X[indexes]
                shapes = self.s[indexes]
                rval.append(numpy.vstack(
                    [self.transformer(img.reshape(s))[None, ...]
                     for img, s in safe_izip(images, shapes)]))
            elif so == 'targets':
                rval.append(self.y[indexes])
        return tuple(rval)

    def get_data_specs(self):
        """
        Returns the data specs for this dataset
        """
        return self.data_specs

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        allowed_modes = ('sequential', 'random_slice', 'even_sequential',
                         'batchwise_shuffled_sequential',
                         'even_batchwise_shuffled_sequential')
        if mode is not None and mode not in allowed_modes:
            raise ValueError("Due to HDF5 limitations on advanced indexing, " +
                             "the '" + mode + "' iteration mode is not " +
                             "supported")

        if data_specs is None:
            data_specs = self._iter_data_specs

        space, source = data_specs
        sub_spaces, sub_sources = (
            (space.components, source) if isinstance(space, CompositeSpace)
            else ((space,), (source,)))
        convert = [None for sp, src in safe_izip(sub_spaces, sub_sources)]

        mode = (self._iter_subset_class if mode is None
                else resolve_iterator_class(mode))

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return VariableImageDatasetIterator(
            dataset=self,
            subset_iterator=mode(
                self.num_examples, batch_size, num_batches, rng),
            data_specs=data_specs,
            return_tuple=return_tuple,
            convert=convert)

    @wraps(Dataset.adjust_for_viewer)
    def adjust_for_viewer(self, X):
        return 2 * X - 1

    @wraps(Dataset.has_targets)
    def has_targets(self):
        return self.y is None

    @wraps(Dataset.get_topo_batch_axis)
    def get_topo_batch_axis(self):
        raise NotImplementedError()

    @wraps(Dataset.get_batch_design)
    def get_batch_design(self, batch_size, include_labels=False):
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_batch_design.")

    @wraps(Dataset.get_batch_topo)
    def get_batch_topo(self, batch_size):
        raise NotImplementedError()

    @wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        raise NotImplementedError()


class BaseImageTransformer(object):
    """
    An object that preprocesses an image on-the-fly
    """
    def get_shape(self):
        """
        Returns the shape of a preprocessed image
        """
        raise NotImplementedError()

    def preprocess(self, image):
        """
        Applies preprocessing on-the-fly

        Parameters
        ----------
        image : numpy.ndarray
            Image to preprocess
        """
        raise NotImplementedError()

    def __call__(self, image):
        return self.preprocess(image)


class RandomCrop(BaseImageTransformer):
    """
    Crops a square at random on a rescaled version of the image

    Parameters
    ----------
    scaled_size : int
        Size of the smallest side of the image after rescaling
    crop_size : int
        Size of the square crop. Must be bigger than scaled_size.
    rng : int or rng, optional
        RNG or seed for an RNG
    """
    _default_seed = 2015 + 1 + 18

    def __init__(self, scaled_size, crop_size, rng=_default_seed):
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        assert self.scaled_size > self.crop_size
        self.rng = make_np_rng(rng, which_method="random_integers")

    @wraps(BaseImageTransformer.get_shape)
    def get_shape(self):
        return (self.crop_size, self.crop_size)

    @wraps(BaseImageTransformer.preprocess)
    def preprocess(self, image):
        small_axis = numpy.argmin(image.shape[:-1])
        ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
        resized_image = misc.imresize(image, ratio)

        max_i = resized_image.shape[0] - self.crop_size
        max_j = resized_image.shape[1] - self.crop_size
        i = self.rng.randint(low=0, high=max_i)
        j = self.rng.randint(low=0, high=max_j)
        cropped_image = resized_image[i: i + self.crop_size,
                                      j: j + self.crop_size, :]
        return cropped_image


class VariableImageDatasetIterator(object):
    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._source = source

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

        for i, (so, sp) in enumerate(safe_izip(source, sub_spaces)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            if init_fn is None:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:
                    fn = (lambda batch, dspace=dspace, sp=sp:
                          dspace.np_format_as(batch, sp))
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn

    def __iter__(self):
        return self

    @wraps(SubsetIterator.next)
    def next(self):
        next_index = self._subset_iterator.next()
        rval = tuple(
            fn(batch) if fn else batch for batch, fn in
            safe_izip(self._dataset.get(self._source, next_index),
                      self._convert)
        )

        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

    def __next__(self):
        return self.next()

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic


if __name__ == "__main__":
    dataset = VariableImageDataset(path='dummy.h5', data_node='Data',
                                   transformer=RandomCrop(20, 15),
                                   X_str='X', y_str='y', s_str='s')
    it = dataset.iterator(mode='random_slice',
                          data_specs=(VectorSpace(dim=675), 'features'),
                          batch_size=10,
                          num_batches=10)
    for X in it:
        print X.shape
