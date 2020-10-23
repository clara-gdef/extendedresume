import argparse
import os
import json
import pickle as pkl
from tqdm import tqdm
import itertools
import yaml
from utils.utils import collate_for_flat_profiles
from torch.utils.data import DataLoader
import ipdb
import fastText
from data.datasets import FlatProfilesDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        load_ds = True

        print("Loading word vectors...")
        ft_edu = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs_edu_job.bin"))
        # ft_pt = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_en.bin"))

        print("Word vectors loaded.")

        for split in ["TEST", "VALID", "TRAIN"]:
            edu_dataset_fs = []
            dataset = FlatProfilesDataset(CFG["gpudatadir"], None, split, None, None, None, None, load_ds)
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate_for_flat_profiles,
                                num_workers=8, shuffle=True)
            for profile in tqdm(loader, desc="Building reps for split " + split):
                embedded_person = {"id": profile[0],
                                   "edu": edu_to_emb(profile[2], ft_edu),
                                   "skills": profile[3],
                                   "ind": profile[4]}
                edu_dataset_fs.append(embedded_person)

            with open(os.path.join(CFG["gpudatadir"], "reps_edu_fs_" + split + ".pkl"), 'wb') as f:
                pkl.dump(edu_dataset_fs)


def edu_to_emb(edu_list, ft_model):
    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
