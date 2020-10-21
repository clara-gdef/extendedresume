import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset


class FlatProfilesDataset(Dataset):
    def __init__(self, input_file, split, ft_job, ft_edu, skills_classes, ind_classes):
        ipdb.set_trace()
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def save_dataset(self, data_dir, agg_type, rep_type, split, standardized):
        pass


