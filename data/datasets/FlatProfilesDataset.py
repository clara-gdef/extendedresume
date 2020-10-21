import json
import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset


class FlatProfilesDataset(Dataset):
    def __init__(self, datadir,  input_file, split, ft_job, ft_edu, skills_classes, ind_classes, load):
        if load:
            self.load_dataset()
        else:
            self.skills_classes = skills_classes
            self.ind_classes = ind_classes
            self.datadir = datadir

            self.tuples = []
            self.build_tuples(input_file, ft_job, ft_edu)
            self.save_dataset()


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass

    def build_tuples(self, input_file, ft_job, ft_edu):
        with open(input_file, 'r') as f:
            for line in f:
                person = json.loads(line)
                ipdb.set_trace()
        pass
