import json
import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils import word_seq_into_list


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
            if sk in self.rev_sk_classes.keys():
                skills_ind.append(self.rev_sk_classes[sk])
        return skills_ind


def handle_jobs(job_list, ft_model):
    new_job_tensor = np.zeros((len(job_list), ft_model.get_dimension()))
    # sort by date, most recent first,
    sorted_jobs = sorted(job_list, key=lambda k: k["from_ts"])
    for num, job in enumerate(sorted_jobs):
        new_job_tensor[num, :] = job_to_emb(job, ft_model)
    return new_job_tensor


def handle_education(edu_list, ft_model):
    new_ed_tensor = np.zeros((len(edu_list), ft_model.get_dimension()))
    for num, edu in enumerate(edu_list):
        tokenized_edu = word_seq_into_list(edu["degree"], edu["institution"])
        word_count = 0
        tmp = []
        for token in tokenized_edu:
            tmp.append(ft_model.get_word_vector(token))
            word_count += 1
        new_ed_tensor[num, :] = np.mean(np.stack(tmp), axis=0) / word_count
    return new_ed_tensor


def job_to_emb(job, ft_model):
    tokenized_jobs = word_seq_into_list(job["position"], job["description"])
    word_count = 0
    emb = np.zeros((1, ft_model.get_dimension()))
    tmp = []
    for token in tokenized_jobs:
        tmp.append(ft_model.get_word_vector(token))
        word_count += 1
    emb[0, :] = np.mean(np.stack(tmp), axis=0) / word_count
    return emb
