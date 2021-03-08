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
from data.datasets import FlatProfilesDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():

        counter_edu = Counter()
        all_edu = []
        counter_jobs = Counter()
        all_jobs = []
        len_edu = []
        len_jobs = []
    
        for item in ["TRAIN", "VALID", "TEST"]: 
            input_file = os.path.join(CFG["gpudatadir"], f"{args.base_file}_{item}.json")
            with open(input_file, 'r') as f:
                num_lines = sum(1 for line in f)
            with open(input_file, 'r') as f:
                pbar = tqdm(f, total=num_lines)
                for line in f:
                    raw_p = json.loads(line)
                    counter_jobs[len(raw_p[1])] += 1
                    for nj, _ in enumerate(raw_p[1]):
                        pos_len = len(raw_p[1][nj]["position"].split(" "))
                        desc_len = len(raw_p[1][nj]["description"].split(" "))
                        len_jobs.append(pos_len + desc_len)
                    all_jobs.append(len(raw_p[1]))
                    counter_edu[len(raw_p[3])] += 1
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
            print(f"Median length of edu sequence : {np.percentile(len_edu, 50)}") #
            print(f"Median length of jobs sequence : {np.percentile(len_jobs, 50)}") #
            print(f"MEAN length of edu sequence :{np.mean(len_edu)}") #
            print(f"MEAN length of jobs sequence : {np.mean(len_jobs)}") #
            print(f" {'='*30}")
            ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
