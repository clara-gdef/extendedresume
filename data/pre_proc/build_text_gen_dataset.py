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
from data.datasets import TextGenerationDataset
from utils.pre_processing import get_ind_class_dict


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    index_file = os.path.join(CFG["gpudatadir"], args.voc_index)
    with open(index_file, 'rb') as f:
        index = pkl.load(f)

    with open(os.path.join(CFG["gpudatadir"], "good_skills.p"), 'rb') as f:
        skills = pkl.load(f)

    skills_classes = {k: v for k, v in enumerate(sorted(skills))}

    ind_classes = get_ind_class_dict(args.build_ind_dict, CFG)

    load = (args.load == "True")

    for split in ["_TEST", "_VALID", "_TRAIN"]:
        input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
        TextGenerationDataset(CFG["gpudatadir"], input_file, index,
                            skills_classes, ind_classes, split, args.ft_type, load)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--voc_index", type=str, default="vocab_40k.pkl")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
