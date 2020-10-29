import argparse
import os
import pickle as pkl
import yaml
import ipdb
from data.datasets import TextGenerationDataset
from utils.pre_processing import get_ind_class_dict


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
        index_file = os.path.join(CFG["gpudatadir"], args.voc_index + args.ft_type + ".pkl")
        with open(index_file, 'rb') as f:
            index = pkl.load(f)
        print("index loaded.")

        ipdb.set_trace()

        load = (args.load_dataset == "True")

        for split in ["_TEST", "_VALID", "_TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            TextGenerationDataset(CFG["gpudatadir"], input_file, index, split, args.ft_type, args.max_seq_length, load)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--build_ind_dict", type=str, default="False")
    parser.add_argument("--voc_index", type=str, default="index_40k_")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
