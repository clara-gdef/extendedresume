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
            fj_ind = prof["first_job"]
            for tok in fj_ind:
                token_count[tok] += 1
        all_tokens = sum(token_count.values()) - token_count[index["PAD"]]
        frequencies = torch.zeros(len(index))
        for tok in token_count.keys():
            frequencies[tok] = all_tokens / token_count[tok]
        freq = 1e-10 + frequencies / all_tokens
        rev_index = {v: k for k, v in index.items()}
        ipdb.set_trace()
        with open(os.path.join(CFG["gpudatadir"], "token_frequencies.pkl"), "wb") as f:
            pkl.dump(freq, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--min_occurence", type=int, default=5)
    args = parser.parse_args()
    main(args)
