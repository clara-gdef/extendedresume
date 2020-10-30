import json
import os
import pickle as pkl
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.pre_processing import word_seq_into_list, word_list_to_indices, handle_education_ft, to_elmo_emb


class TextGenerationDataset(Dataset):
    def __init__(self, datadir, input_file, index, split, ft_type, max_seq_length, embedder, load):
        if load:
            self.datadir = datadir
            self.load_dataset(split, ft_type)
        else:
            self.max_seq_length = max_seq_length
            self.index = index
            self.ft_type = ft_type

            self.datadir = datadir

            self.tuples = []
            self.build_tuples(input_file, index, max_seq_length, embedder, split)
            self.save_dataset(split, ft_type)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"],  \
               self.tuples[idx]["edu_" + self.ft_type], \
               self.tuples[idx]["first_job"], \
               self.tuples[idx]["job_len"]

    def save_dataset(self, split, ft_type):
        dico = {"datadir": self.datadir,
                'max_seq_length': self.max_seq_length,
                "index": self.index,
                "ft_type": ft_type,
                "tuples": self.tuples}
        with open(os.path.join(self.datadir, "text_gen_dataset_" + ft_type + split + ".pkl"), 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self, split, ft_type):
        with open(os.path.join(self.datadir, "text_gen_dataset_" + ft_type + "_" + split + ".pkl"), 'rb') as f:
            dico = pkl.load(f)
        self.ft_type = dico["ft_type"]
        self.datadir = dico["datadir"]
        ##################
        np.random.shuffle(dico["tuples"])
        # self.tuples = dico["tuples"][:1000]
        self.tuples = dico["tuples"]
        self.index = dico["index"]
        ###########
        print("Data length: " + str(len(self.tuples)))

    def build_tuples(self, json_file, index, max_seq_length, embedder, split):
        with open(json_file, 'r') as f:
            num_lines = sum(1 for line in f)
        with open(json_file, 'r') as f:
            pbar = tqdm(f, total=num_lines, desc="Building text_gen dataset for split " + split + " & " + self.ft_type + " ...")
            for line in f:
                data = json.loads(line)
                job_list = sorted(data[1], key=lambda k: k["from_ts"], reverse=True)
                tokenized_first_job = word_seq_into_list(job_list[-1]["position"], job_list[-1]["description"], index)
                if len(data[-2]) > 0:
                    if self.ft_type != "elmo":
                        new_edu = handle_education_ft(data[-2], embedder)
                        first_job, job_len = word_list_to_indices(tokenized_first_job, index, max_seq_length)
                    else:
                        new_edu = to_elmo_emb(data[-2], embedder)
                        first_job = tokenized_first_job
                        job_len = len(first_job)
                    self.tuples.append({
                        "id": data[0],
                        "edu_" + self.ft_type: new_edu,
                        "first_job": first_job,
                        "job_len": job_len
                    })
                pbar.update(1)

    def handle_skills(self, skill_list):
        skills_ind = []
        for sk in skill_list:
            if sk in self.rev_sk_classes.keys():
                skills_ind.append(self.rev_sk_classes[sk])
        return skills_ind
