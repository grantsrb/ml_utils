from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import random

MASK = "<MASK>"
STOP = "<STOP>"
START = "<START>"
FILL = "<FILL>"

class ListWrapper:
    def __init__(self, arr, shape=None):
        """
        arr: some list
        shape: a tuple that does not include the len of the arr
        """
        self.arr = arr
        self._shape = tuple(shape)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self,idx):
        try:
            return self.arr[idx]
        except:
            temp = []
            for i in idx:
                temp.append(self.arr[i])
            return temp

    @property
    def shape(self):
        shape = tuple([len(self.arr)]+list(self._shape))
        return shape

class DatasetSplitWrapper(Dataset):
    """
    Used as a wrapper class to more easily split a dataset into a
    validation and training set
    """
    def __init__(self,dataset,idxs):
        """
        dataset: torch Dataset
        idxs: torch LongTensor or list of ints
        """
        self.dataset = dataset
        for attr in dir(dataset):
            if "__"!=attr[:2]:
                try:
                    setattr(self, attr, getattr(dataset,attr))
                except:
                    pass
        self.idxs = idxs
        assert len(self.idxs) <= len(self.dataset)
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        try:
            idx = self.idxs[idx]
            return self.dataset[idx]
        except FileNotFoundError as e:
            while True:
                try:
                    idx = rand_sample(self.idxs)
                    return self.dataset[idx]
                except FileNotFoundError as e:
                    pass

