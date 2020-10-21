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
            self.rev_sk_classes = {v: k for k, v in skills_classes.items()}
            self.ind_classes = ind_classes
            self.rev_ind_classes = {v: k for k, v in ind_classes.items()}

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
            num_lines = sum(1 for line in f)
        with open(input_file, 'r') as f:
            pbar = tqdm(f, total=num_lines)
            for line in f:
                raw_p = json.loads(line)
                ipdb.set_trace()
                self.tuples.append({
                    "id": raw_p[0],
                    "jobs": handle_jobs(raw_p[1], ft_job),
                    "skills": self.handle_skills(raw_p[2]),
                    "edu": handle_education(raw_p[3], ft_edu),
                    "ind": self.rev_ind_classes[raw_p[4]]
                })
            pbar.update(1)

    def handle_skills(self, skill_list):
        skills_ind = []
        for sk in skill_list:
            skills_ind.append(self.rev_sk_classes[sk])
        return skills_ind


def handle_jobs(job_list, ft_model):
    # sort by date, most recent first,
    # stadardize (??)
    # compute rep for each job for each profile
    ipdb.set_trace()


def handle_education(edu_list, ft_model):
    # sort by date, most recent first,

    ipdb.set_trace()
