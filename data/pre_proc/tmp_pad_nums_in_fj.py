import argparse
import os
import pickle as pkl

from tqdm import tqdm
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

        len_fj = []
        for split in ["TEST", "VALID", "TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            dataset = TextGenerationDataset(CFG["gpudatadir"], input_file, None, split, args.ft_type, args.max_seq_length, None, 0, True)
            for tup in tqdm(dataset, desc="Parsing dataset for split " + split):
                len_fj.append(tup[-1])


        for split in ["TEST", "VALID", "TRAIN"]:
            num_tokens = 0
            num_pad = 0
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + split + ".json")
            dataset = TextGenerationDataset(CFG["gpudatadir"], input_file, None, split, args.ft_type, args.max_seq_length, None, 0, True)
            for tup in tqdm(dataset, desc="Parsing dataset for split " + split):
                for tok in tup[-2]:
                    if tok == index["PAD"]:
                        num_pad += 1
                    num_tokens += 1
            print(str(num_pad*100/num_tokens) + "% of tokens are PAD tokens")
            ipdb.set_trace()

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
