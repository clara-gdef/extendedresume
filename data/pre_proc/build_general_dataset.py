import argparse
import os
import json
import pickle as pkl
from tqdm import tqdm
import itertools
import yaml
import ipdb
import fastText
from data.datasets import FlatProfilesDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(CFG["gpudatadir"], "good_skills.p"), 'rb') as f:
            skills = pkl.load(f)

        skills_classes = {k: v for k, v in enumerate(sorted(skills))}

        ind_classes = get_ind_class_dict(args)

        load_ds = (args.load_dataset == "True")

        print("Loading word vectors...")
        ft_edu = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs_edu_job.bin"))
        ft_jobs = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs.bin"))
        print("Word vectors loaded.")

        for split in ["_TEST", "_VALID", "_TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            FlatProfilesDataset(CFG["gpudatadir"], input_file, split, ft_jobs, ft_edu, skills_classes, ind_classes, load_ds)


def get_ind_class_dict(args):
    if args.build_ind_dict == "True":
        class_dict = build_ind_class_dict()
        with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'wb') as f:
            pkl.dump(class_dict, f)
    else:
        with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f:
            class_dict = pkl.load(f)
    return class_dict


def build_ind_class_dict():
    print("Building industry class dict...")
    input_files = []
    for split in ["_TEST", "_VALID", "_TRAIN"]:
        input_files.append(os.path.join(CFG["prevdatadir"], args.base_file + split + ".json"))

    classes = set()
    for filename in itertools.chain(input_files):
        with open(filename, "r") as f:
            for line in f:
                person = json.loads(line)
                classes.add(person[-1])

    class_dict = {}
    for num, industry in enumerate(sorted(classes)):
        class_dict[num] = industry
    print("Done.")
    return class_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)