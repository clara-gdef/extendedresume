import argparse
import os
import json
import pickle as pkl
from tqdm import tqdm
import itertools
import yaml
import ipdb
from allennlp.modules.elmo import Elmo
import fastText
from data.datasets import FlatProfilesDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    input_file = os.path.join(CFG["gpudatadir"], args.base_file + "_TRAIN.json")
    with open(input_file, 'r') as f:
        num_lines = sum(1 for line in f)
    with open(input_file, 'r') as f:
        pbar = tqdm(f, total=num_lines, desc="Building vocab from Train split...")
        for line in f:
            data = json.loads(line)




            
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--voc_index", type=str, default="vocab_40k.pkl")
    parser.add_argument("--elmo", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
