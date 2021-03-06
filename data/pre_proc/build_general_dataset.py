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
from utils.pre_processing import get_ind_class_dict


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(CFG["gpudatadir"], "good_skills.p"), 'rb') as f:
            skills = pkl.load(f)

        skills_classes = {k: v for k, v in enumerate(sorted(skills))}

        ind_classes = get_ind_class_dict(args.build_ind_dict, CFG)

        load_ds = (args.load_dataset == "True")
        elmo = (args.elmo == "True")

        print("Loading word vectors...")
        if elmo:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            elmo = Elmo(options_file, weight_file, 2, dropout=0)
            elmo = elmo.cuda()
            ft_edu, ft_jobs, ft_pt = None, None, None
        else:
            elmo = None
            ft_edu = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs_edu_job.bin"))
            ft_jobs = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs.bin"))
            ft_pt = fastText.load_model(os.path.join(CFG["modeldir"], "ft_pre_trained.bin"))
        print("Word vectors loaded.")

        for split in ["_TEST", "_VALID", "_TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            FlatProfilesDataset(CFG["gpudatadir"], input_file, split, ft_jobs, ft_edu, ft_pt, elmo,
                                skills_classes, ind_classes, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--elmo", type=str, default="False")
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
