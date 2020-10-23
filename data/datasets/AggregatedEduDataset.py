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


class AggregatedEduDataset(Dataset):
    def __init__(self, datadir, input_file, split, ft_type, load):
        if load:
            self.datadir = datadir
            self.load_dataset(split, ft_type)
        else:
            with open(input_file, "rb") as f:
                general_ds = pkl.load(f)

            self.ft_type = ft_type
            self.skills_classes = general_ds["skills_classes"]
            self.rev_sk_classes = general_ds["rev_sk_classes"]
            self.ind_classes = general_ds["ind_classes"]
            self.rev_ind_classes = general_ds["rev_ind_classes"]

            self.datadir = datadir

            self.tuples = []
            self.build_tuples(general_ds, ft_type)
            self.save_dataset(split, ft_type)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"],  \
               self.tuples[idx]["edu"], \
               self.tuples[idx]["skills"], \
               self.tuples[idx]["ind"]

    def save_dataset(self, split, ft_type):
        dico = {"skills_classes": self.skills_classes,
                "rev_sk_classes": self.rev_sk_classes,
                "ind_classes": self.ind_classes,
                "rev_ind_classes": self.rev_ind_classes,
                "datadir": self.datadir,
                "tuples": self.tuples}
        with open(os.path.join(self.datadir, "agg_edu_dataset_" + ft_type + "_" + split + ".pkl"), 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, split, ft_type):
        with open(os.path.join(self.datadir, "agg_edu_dataset_" + ft_type + "_" + split + ".pkl"), 'rb') as f:
            dico = pkl.load(f)
        self.skills_classes = dico["skills_classes"]
        self.rev_sk_classes = dico["rev_sk_classes"]
        self.ind_classes = dico["ind_classes"]
        self.rev_ind_classes = dico["rev_ind_classes"]
        self.datadir = dico["datadir"]
        ##################
        np.random.shuffle(dico["tuples"])
        # self.tuples = dico["tuples"][:1000]
        self.tuples = dico["tuples"]
        ###########
        print("Data length: " + str(len(self.tuples)))

    def build_tuples(self, general_ds, ft_type):
        for person in tqdm(general_ds["tuples"], desc="Parse general dataset..."):
            new_p = {}
            for key in ["id", "ind", "skills"]:
                new_p[key] = person[key]
            new_p["edu_" + ft_type] = to_avg_emb(person["edu"])
            self.tuples.append(new_p)


def to_avg_emb(emb_list):
    ipdb.set_trace()
    return torch.FloatTensor(torch.from_numpy(np.mean(emb_list, axis=0)))


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
