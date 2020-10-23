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

        input_file = os.path.join(CFG["gpudatadir"], args.base_file + "_TRAIN.json")
        with open(input_file, 'r') as f:
            num_lines = sum(1 for line in f)
        with open(input_file, 'r') as f:
            pbar = tqdm(f, total=num_lines)
            for line in f:
                raw_p = json.loads(line)
                counter_jobs[len(raw_p[1])] += 1
                all_jobs.append(len(raw_p[1]))
                counter_edu[len(raw_p[3])] += 1
                all_edu.append(len(raw_p[3]))
                pbar.update(1)
        print(str(np.percentile(all_edu, 90))) # 4
        print(str(np.percentile(all_jobs, 90))) # 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)