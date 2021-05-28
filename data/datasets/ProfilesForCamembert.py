import json
import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.pre_processing import word_seq_into_list


class ProfilesForCamembert(Dataset):
    def __init__(self, datadir, input_file, split, skills_classes, ind_classes, subsample, load):
        self.datadir = datadir
        self.split = split
        self.skills_classes = skills_classes
        self.rev_sk_classes = {v: k for k, v in skills_classes.items()}
        self.ind_classes = ind_classes
        self.rev_ind_classes = {v: k for k, v in ind_classes.items()}
        self.name = "ProfilesForCamembert"
        if load:
            self.load_dataset(subsample)
        else:
            self.build_tuples(input_file, split)
            self.save_dataset(subsample)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"],  \
               self.tuples[idx]["jobs"], \
               self.tuples[idx]["edu"], \
               self.tuples[idx]["skills"], \
               self.tuples[idx]["ind"]

    def save_dataset(self, subsample):
        print("Subsampling dataset...")
        np.random.shuffle(self.tuples)
        retained_tuples = self.tuples[:subsample]
        self.tuples = retained_tuples
        print(f"len tuples in save_dataset: {len(self.tuples)}")
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = self.get_tgt_file(subsample)

        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, subsample):
        tgt_file = self.get_tgt_file(subsample)
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)

        for key in tqdm(ds_dict, desc="Loading attributes from save..."):
            vars(self)[key] = ds_dict[key]
        print("Dataset load from : " + tgt_file)

    def build_tuples(self, input_file, split):
        self.tuples = []
        with open(input_file, 'r') as f:
            num_lines = sum(1 for line in f)
        with open(input_file, 'r') as f:
            pbar = tqdm(f, total=num_lines, desc="Building tuple for split: " + split)
            for line in f:
                raw_p = json.loads(line)
                new_p = {
                    "id": raw_p[0],
                    "jobs": handle_jobs(raw_p[1]),
                    "edu": handle_education(raw_p[3]),
                    "skills": self.handle_skills(raw_p[2]),
                    "ind": self.rev_ind_classes[raw_p[4]]
                }
                flag = True
                for k in ["edu", "jobs", "skills"]:
                    if len(new_p[k]) < 1:
                        flag = False
                if flag:
                    self.tuples.append(new_p)
                pbar.update(1)

    def handle_skills(self, skill_list):
        skills_ind = []
        for sk in skill_list:
            if sk in self.rev_sk_classes.keys():
                skills_ind.append(self.rev_sk_classes[sk])
        return skills_ind

    def get_tgt_file(self, subsample):
        if subsample != -1:
            tgt_file = os.path.join(self.datadir, f"{self.name}_{self.split}_{subsample}.pkl")
        else:
            tgt_file = os.path.join(self.datadir, f"{self.name}_{self.split}.pkl")
        return tgt_file


def handle_jobs(job_list):
    new_job_list = []
    # keeps 90% of the dataset without trimming experience
    max_jobs = 8
    # sort by date, most recent first,
    sorted_jobs = sorted(job_list, key=lambda k: k["from_ts"], reverse=True)
    for num, job in enumerate(sorted_jobs):
        if num < max_jobs:
            new_job_list.append(job["position"] + " " + job["description"])
    return new_job_list


def handle_education(edu_list):
    new_edu_list = []
    sorted_edu_list = sorted(edu_list, key=lambda k: k["to"], reverse=True)
    # keeps 90% of the dataset without trimming experience
    max_edu = 4
    for num, edu in enumerate(sorted_edu_list):
        if num < max_edu:
            new_edu_list.append(edu["degree"] + " " + edu["institution"])
    return new_edu_list



