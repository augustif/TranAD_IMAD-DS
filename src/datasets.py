import h5py
import torch
from torch.utils.data import Dataset

class LazyHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, transform=None, dataset_name='train', n_win_per_chunk= 2048):
        self.h5_file = h5_file_path
        self.transform = transform
        self.labels = None #complete labels vector
        self.single_label = None # single sample label
        self.sample_size = 1600 #100 ms window
        self.n_win_per_chunk = n_win_per_chunk
        self.chunk_size = self.sample_size*self.n_win_per_chunk
        self.load_idx = 0

        # Open the HDF5 file to get the length of the dataset
        with h5py.File(self.h5_file,'r') as h5_file:
            if dataset_name in h5_file.keys():
                self.dataset_name = dataset_name
                self.data_len = len(h5_file[dataset_name])
                if dataset_name == 'test':
                    self.labels = torch.tensor(h5_file['labels'][:])
                else:
                    self.labels = None #torch.zeros(h5_file[dataset_name].shape)

    def __len__(self):
        return int(self.data_len/self.chunk_size)

    def get_chunk(self):
        return self.__getitem__(self.load_idx)

    def __getitem__(self, idx):
        
        with h5py.File(self.h5_file,'r') as h5_file:
            sample = h5_file[self.dataset_name][idx*self.chunk_size:(idx+1)*self.chunk_size, :]
            if self.dataset_name == 'test':
                self.single_label = h5_file['labels'][idx]
            elif self.dataset_name == 'train': 
                self.single_label = None #torch.zeros(len(sample))

            if self.transform:
                sample = self.transform(sample)

            self.load_idx += self.chunk_size
        return sample
    
    # def close(self):
    #     if self.h5_file:
    #         self.h5_file.close()
    #         print("HDF5 file closed.")

    # def __del__(self):
    #     self.close()

class LazyHDF5Dataset_windowed(Dataset):
    def __init__(self, h5_file_path, transform=None, dataset_name='train', n_win_per_chunk= 2048):

        self.dataloader = LazyHDF5Dataset(h5_file_path, dataset_name=dataset_name, n_win_per_chunk=n_win_per_chunk)
        self.loaded_chunk= None
        self.transform = transform

    def __len__(self):
        return self.dataloader.len

    def __getitem__(self, idx):
        if (self.loaded_chunk is None) or(idx >= len(self.loaded_chunk)):
            self.loaded_chunk = self.dataloader.get_chunk()

        return self.loaded_chunk[idx,:,:]