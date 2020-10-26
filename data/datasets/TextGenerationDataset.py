import json
import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.model import word_seq_into_list


class TextGenerationDataset(Dataset):
    def __init__(self, datadir, input_file, index, skills_classes, ind_classes, split, ft_type, load):
        if load:
            self.datadir = datadir
            self.load_dataset(split, ft_type)
        else:
            with open(input_file, "rb") as f:
                json_file = pkl.load(f)

            self.index = index
            self.ft_type = ft_type
            self.skills_classes = skills_classes
            self.rev_sk_classes = {v: k for k, v in skills_classes.items()}
            self.ind_classes = ind_classes
            self.rev_ind_classes = {v: k for k, v in ind_classes.items()}

            self.datadir = datadir

            self.tuples = []
            self.build_tuples(json_file, index, ft_type)
            self.save_dataset(split, ft_type)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"],  \
               self.tuples[idx]["edu_" + self.ft_type], \
               self.tuples[idx]["skills"], \
               self.tuples[idx]["ind"]

    def save_dataset(self, split, ft_type):
        dico = {"skills_classes": self.skills_classes,
                "rev_sk_classes": self.rev_sk_classes,
                "ind_classes": self.ind_classes,
                "rev_ind_classes": self.rev_ind_classes,
                "datadir": self.datadir,
                "ft_type": ft_type,
                "tuples": self.tuples}
        with open(os.path.join(self.datadir, "text_gen_dataset_" + ft_type + "_" + split + ".pkl"), 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, split, ft_type):
        with open(os.path.join(self.datadir, "text_gen_dataset_" + ft_type + "_" + split + ".pkl"), 'rb') as f:
            dico = pkl.load(f)
        self.skills_classes = dico["skills_classes"]
        self.rev_sk_classes = dico["rev_sk_classes"]
        self.ind_classes = dico["ind_classes"]
        self.ft_type = dico["ft_type"]
        self.rev_ind_classes = dico["rev_ind_classes"]
        self.datadir = dico["datadir"]
        ##################
        np.random.shuffle(dico["tuples"])
        # self.tuples = dico["tuples"][:1000]
        self.tuples = dico["tuples"]
        ###########
        print("Data length: " + str(len(self.tuples)))

    def build_tuples(self, json_file, index, ft_type):
        with open(json_file, 'r') as f:
            num_lines = sum(1 for line in f)
        with open(json_file, 'r') as f:
            pbar = tqdm(f, total=num_lines, desc="Building vocab from Train split...")
            for line in f:
                data = json.loads(line)
                edu_list = sorted(data[-2], key=lambda k: k["to"], reverse=True)
                for edu in edu_list:
                    splitted_sentence = word_seq_into_list(edu["degree"], edu["institution"])
                    ipdb.set_trace()

                pbar.update(1)
