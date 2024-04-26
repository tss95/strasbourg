import torch
from torch.utils.data import Dataset
import os
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from time import time


class AvaPretrainDataset(Dataset):
    
    def __init__(self, data_path, dataset_type, mfr_transform=None, trace_transform=None, image_size = (256, 256), seed=42):
        self.data_path = data_path
        self.trace_transform = trace_transform
        self.mfr_transform = mfr_transform
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.seed = seed
        self.paths = os.listdir(data_path)
        self.paths = self.__split_paths()
        self.min_size = None

    def __split_paths(self):
        train_paths, test_paths = train_test_split(self.paths, test_size=0.2, random_state=self.seed)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=self.seed)  # 0.25 x 0.8 = 0.2
        if self.dataset_type == 'train':
            return train_paths
        elif self.dataset_type == 'val':
            return val_paths
        elif self.dataset_type == 'test':
            return test_paths
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        #start = time()
        f = h5py.File(os.path.join(self.data_path, path), 'r')
        mfr_raw = f['mfr'][:]
        max_trace_raw = f['max_trace'][:]
        not_nan_mask = ~np.isnan(mfr_raw).any(axis=(0))
        mfr_raw = mfr_raw[:, not_nan_mask]
        
        if self.min_size is None or max_trace_raw.shape[0] < self.min_size:
            self.min_size = max_trace_raw.shape[0]
        mfr1 = mfr_raw
        mfr2 = mfr_raw
        max_trace1 = max_trace_raw
        max_trace2 = max_trace_raw
        #print(f"Time to load data: {round(time() - start, 4)}")
        # if self.trace_transform:
        #     # Print to understand how long this takes:
        #     start = time()
        #     max_trace1 = self.trace_transform(max_trace1)
        #     max_trace2 = self.trace_transform(max_trace2)
        if self.trace_transform:
            start = time()
            for transform in self.trace_transform.transforms:
                start_transform = time()
                max_trace1 = transform(max_trace1)
                max_trace2 = transform(max_trace2)
                #print(f"Trace transform {transform.__class__.__name__} time: {round(time() - start_transform, 4)}")
            #print(f"Total Trace transform time: {round(time() - start, 4)}")
        # if self.mfr_transform:
        #     start = time()
        #     mfr1 = self.mfr_transform(mfr1)
        #     mfr2 = self.mfr_transform(mfr2)
        if self.mfr_transform:
            start = time()
            for transform in self.mfr_transform.transforms:
                start_transform = time()
                mfr1 = transform(mfr1)
                mfr2 = transform(mfr2)
                #print(f"MFR transform {transform.__class__.__name__} time: {round(time() - start_transform, 4)}")
            #print(f"Total MFR transform time: {round(time() - start, 4)}")
        
        mfr1 = torch.tensor(mfr1, dtype=torch.float32)
        mfr2 = torch.tensor(mfr2, dtype=torch.float32)
        max_trace1 = torch.tensor(max_trace1, dtype=torch.float32)
        max_trace2 = torch.tensor(max_trace2, dtype=torch.float32)

        return {"mfr1": mfr1, "mfr2": mfr2, "max_trace1": max_trace1, "max_trace2": max_trace2}
    
                    