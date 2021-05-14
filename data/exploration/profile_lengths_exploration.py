import argparse
import os
import json
import pickle as pkl
from tqdm import tqdm
import itertools
import numpy as np
import yaml
from collections import Counter
import ipdb
import fastText
from data.datasets import FlatProfilesDataset, TextGenerationDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():

        # counter_edu = Counter()
        # counter_jobs = Counter()
        all_edu, all_jobs, len_edu, len_jobs = [], [], [], []
        total_edu = 0; total_job = 0
        total_profiles = {}
        for item in ["TRAIN", "VALID", "TEST"]:
            total_profiles[item] = 0
            if args.load_dataset == "False":
                input_file = os.path.join(CFG["gpudatadir"], f"{args.base_file}_{item}.json")
                with open(input_file, 'r') as f:
                    num_lines = sum(1 for line in f)
                with open(input_file, 'r') as f:
                    pbar = tqdm(f, total=num_lines, desc=f"Parsing for split {item}...")
                    for line in f:
                        raw_p = json.loads(line)
                        # counter_jobs[len(raw_p[1])] += 1
                        for nj, _ in enumerate(raw_p[1]):
                            pos_len = len(raw_p[1][nj]["position"].split(" "))
                            desc_len = len(raw_p[1][nj]["description"].split(" "))
                            len_jobs.append(pos_len + desc_len)
                        all_jobs.append(len(raw_p[1]))
                        # counter_edu[len(raw_p[3])] += 1
                        for ne, _ in enumerate(raw_p[3]):
                            deg_len = len(raw_p[3][ne]["degree"].split(" "))
                            inst_len = len(raw_p[3][ne]["institution"].split(" "))
                            len_edu.append(deg_len + inst_len)
                        all_edu.append(len(raw_p[3]))
                        pbar.update(1)
                print(f"NUM OCCURENCE FOR {item} SPLIT")
                # print(f"90 percentile num edu per profile : {np.percentile(all_edu, 90)}") # 4
                # print(f"90 percentile num jobs per profile : {np.percentile(all_jobs, 90)}") # 8
                print(f"MEDIAN num edu per profile : {np.percentile(all_edu, 50)}") #
                print(f"MEDIAN percentile num jobs per profile : {np.percentile(all_jobs, 50)}") #
                print(f"MEAN num edu per profile : {np.mean(all_edu)}") # 4
                print(f"MEAN num jobs per profile : {np.mean(all_jobs)}") # 8
                print(f" {'='*10}")
                print(f" SEQUENCE LENGTHS FOR {item} SPLIT")
                print(f"NUM OF EDUCATION : {len(len_edu)}") #
                print(f"NUM OF JOBS : {len(len_jobs)}") #
                print(f"Median length of edu sequence : {np.percentile(len_edu, 50)}") #
                print(f"Median length of jobs sequence : {np.percentile(len_jobs, 50)}") #
                print(f"MEAN length of edu sequence :{np.mean(len_edu)}") #
                print(f"MEAN length of jobs sequence : {np.mean(len_jobs)}") #
                print(f" {'='*30}")
                total_edu += len(len_edu)
                total_job += len(len_jobs)
                print(f"NUM OF EDUCATION : {total_edu}")  #
                print(f"NUM OF JOBS : {total_job}")  #
            else:
                common_hparams = {
                    "input_file": None,
                    "index": None,
                    "embedder": None,
                    "max_seq_length": None,
                    "datadir": CFG["gpudatadir"],
                    "ft_type": "fs",
                    "load": (args.load_dataset == "True"),
                    "subsample": 0
                }
                edu_len = []
                dataset = TextGenerationDataset(**common_hparams, split=item)
                for person in dataset.tuples:
                    total_profiles[item] += 1
                    edu_len.append(person["job_len"])
        ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
