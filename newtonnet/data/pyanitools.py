# Written by Roman Zubatyuk and Justin S. Smith
import h5py
import numpy as np
import platform
import os


PY_VERSION = int(platform.python_version().split('.')[0]) > 3

class datapacker(object):
    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6):
        """Wrapper to store arrays within HFD5 file
        """
        # opening file
        self.store = h5py.File(store_file, mode=mode)
        self.clib = complib
        self.clev = complevel

    def store_data(self, store_loc, **kwargs):
        """Put arrays to store
        """
        #print(store_loc)
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            #print(type(v[0]))

            #print(k)
            if type(v) == list:
                if len(v) != 0:
                    if type(v[0]) is np.str_ or type(v[0]) is str:
                        v = [a.encode('utf8') for a in v]

            g.create_dataset(k, data=v, compression=self.clib, compression_opts=self.clev)

    def cleanup(self):
        """Wrapper to close HDF5 file
        """
        self.store.close()


class anidataloader(object):

    ''' Contructor '''
    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - '+store_file)
        self.store = h5py.File(store_file)

    ''' Group recursive iterator (iterate through all groups in all branches and return datasets in dicts) '''
    def h5py_dataset_iterator(self,g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset): # test for dataset
                data = {'path':path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        if int(h5py.__version__.split('.')[0]) >= 3: # check h5py version
                            dataset = item[k][()]
                        else:
                            dataset = np.array(item[k].value)

                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii') for a in dataset]

                        data.update({k:dataset})

                yield data
            else: # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        for data in self.h5py_dataset_iterator(g):
            yield data

    ''' Returns the requested dataset '''
    def get_data(self, path, prefix=''):
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        # print(path)
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                if int(h5py.__version__.split('.')[0]) >= 3: # check h5py version
                    dataset = item[k][()]
                else:
                    dataset = np.array(item[k].value)

                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    ''' Returns the number of groups '''
    def group_size(self):
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()


class DataLoaderAniccx(object):

    ''' Contructor '''
    def __init__(self, store_file, keys):
        if not os.path.exists(store_file):
            exit('Error: file not found - '+store_file)
        self.store = h5py.File(store_file)
        self.file_name = store_file
        if keys == "original": # Original ANI-1x data
            self.keys = ['wb97x_dz.energy','wb97x_dz.forces']
        elif keys == "CHNO": # CHNO portion of the data set used in AIM-Net
            self.keys = ['wb97x_tz.energy','wb97x_tz.forces']
        elif keys == "ccsd": # The coupled cluster ANI-1ccx data set
            self.keys = ['ccsd(t)_cbs.energy']
        elif keys == "dipole": # A subset of this data was used for training the ACA charge model
            self.keys == ['wb97x_dz.dipoles']
        else:
            print("Error: No matching keys. Check github readme for help.")

    def iter_data_buckets(self, h5filename):
        """ Iterate over buckets of data in ANI HDF5 file. 
        Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
        and other available properties specified by `keys` list, w/o NaN values.
        """
        keys = set(self.keys)
        keys.discard('atomic_numbers')
        keys.discard('coordinates')
        with h5py.File(h5filename, 'r') as f:
            for grp in f.values():
                Nc = grp['coordinates'].shape[0]
                mask = np.ones(Nc, dtype=np.bool_)
                data = dict((k, grp[k][()]) for k in keys)
                for k in keys:
                    v = data[k].reshape(Nc, -1)
                    mask = mask & ~np.isnan(v).any(axis=1)
                if not np.sum(mask):
                    continue
                d = dict((k, data[k][mask]) for k in keys)
                d['atomic_numbers'] = grp['atomic_numbers'][()]
                d['coordinates'] = grp['coordinates'][()][mask]
                yield d 


    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        # for data in self.h5py_dataset_iterator(self.store):
        #     yield data
        #print("working new iter")
        for data in self.iter_data_buckets(self.file_name):
            yield data

    ''' Returns the avaliable energy type '''
    def get_energy_type(self):
        return self.keys[0]

    ''' Returns the avaliable energy type '''
    def get_force_type(self):
        return self.keys[1]
    ''' Returns the number of groups '''
    def group_size(self):
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()
