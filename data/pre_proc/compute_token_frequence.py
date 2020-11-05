import argparse
import os
import pickle as pkl
import yaml
import torch
import ipdb
from collections import Counter
from tqdm import tqdm

from data.datasets import TextGenerationDataset


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        index_file = os.path.join(CFG["gpudatadir"], "index_40k.pkl")
        with open(index_file, 'rb') as f:
            index = pkl.load(f)
        ds_train = TextGenerationDataset(CFG["gpudatadir"], None, None, "TRAIN", args.ft_type, None, None, 0, True)
        token_count = Counter()
        for prof in tqdm(ds_train.tuples, desc="Parsong dataset..."):
            fj_ind = prof[2]
            for tok in fj_ind:
                if tok != index["PAD"]:
                    token_count[tok] += 1
        ipdb.set_trace()
        all_tokens = sum(token_count.values())
        frequencies = torch.zeros(1, len(index))
        for num, tok in enumerate(token_count.keys()):
            frequencies[:, num] = token_count[tok] / all_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--min_occurence", type=int, default=5)
    args = parser.parse_args()
    main(args)
