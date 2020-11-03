import argparse
import os
import pickle as pkl
from allennlp.modules.elmo import Elmo
import fastText
import yaml
import ipdb
from data.datasets import TextGenerationDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
        index_file = os.path.join(CFG["gpudatadir"], args.voc_index + args.ft_type + ".pkl")
        with open(index_file, 'rb') as f:
            index = pkl.load(f)
        print("index loaded.")


        for split in ["_TEST", "_VALID", "_TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            dataset = TextGenerationDataset(CFG["gpudatadir"], input_file, index, split, args.ft_type, args.max_seq_length, None, 0, True)
            dataset.fj_to_ind_for_elmo(split)
            dataset.save()


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
