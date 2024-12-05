import h5py
import torch
from torch.utils.data import Dataset

class LazyHDF5DatasetChunks(Dataset):
    def __init__(self, h5_file_path, transform=None, dataset_name='train', n_win_per_chunk= 2048):
        self.h5_file = h5_file_path
        self.transform = transform
        self.data_len = None # actual data length
        self.labels = None #complete labels vector
        self.single_label = None # single sample label
        self.sample_size = 1600 #100 ms window
        self.n_win_per_chunk = n_win_per_chunk
        self.chunk_size = self.sample_size*self.n_win_per_chunk
        self.load_idx = 0 # i-th chunk

        # Open the HDF5 file to get the length of the dataset
        with h5py.File(self.h5_file,'r') as h5_file:
            if dataset_name in h5_file.keys():
                self.dataset_name = dataset_name
                self.data_len = len(h5_file[dataset_name])
                if dataset_name == 'test':
                    self.labels = torch.tensor(h5_file['labels'][:])

    def get_len(self):
        return self.__len__()
    
    def __len__(self):
        # will stop after reading all chunks in h5
        return int(self.data_len/self.chunk_size)

    def get_chunk(self):
        chunk = self.__getitem__(self.load_idx)
        labels = self.single_label
        self.load_idx += 1
        return chunk, labels

    def __getitem__(self, idx):
        
        with h5py.File(self.h5_file,'r') as h5_file:
            sample = h5_file[self.dataset_name][idx*self.chunk_size:(idx+1)*self.chunk_size, :]
            if self.dataset_name == 'test':
                self.single_label = h5_file['labels'][idx*self.chunk_size:(idx+1)*self.chunk_size]

            if self.transform:
                sample = self.transform(sample)
        return sample

class LazyHDF5DatasetWindows(Dataset):
    def __init__(self, h5_file_path, transform=None, dataset_name='train', n_win_per_chunk= 2048):

        self.dataset = LazyHDF5DatasetChunks(h5_file_path, dataset_name=dataset_name, n_win_per_chunk=n_win_per_chunk)
        self.chunk= None
        self.chunk_labels = None
        self.transform = transform
        self.idx = 0

        if dataset_name == 'train':
            with h5py.File(h5_file_path,'r') as h5_file:
                if dataset_name in h5_file.keys():
                    feats, channels = h5_file['train'][0].shape
                    self.dataset.single_label = torch.zeros((self.dataset.chunk_size, feats, channels))

    def __len__(self):
        # will stop after reading all samples read trhough iteration over the dataset
        return self.dataset.get_len() * self.dataset.sample_size

    def __getitem__(self, _):
        if (self.chunk is None) or(self.idx >= len(self.chunk)-1):
            self.chunk, self.chunk_labels = self.dataset.get_chunk()
            self.idx = 0
        sample, label = self.chunk[self.idx], self.chunk_labels[self.idx]
        self.idx += 1
        return sample, label