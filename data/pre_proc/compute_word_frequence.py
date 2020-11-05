import os
import pickle as pkl
import yaml
import ipdb
from tqdm import tqdm
from utils import build_word_set


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    # small sanity check
    index = []
    for ft_type in tqdm(["fs", "pt", "elmo"]):
        index_file = os.path.join(CFG["gpudatadir"], "index_40k_" + ft_type + ".pkl")
        with open(index_file, 'rb') as f:
            index.append(pkl.load(f))
    assert index[0] == index[1]
    assert index[2] == index[1]

    with ipdb.launch_ipdb_on_exception():
        input_file = os.path.join(CFG["gpudatadir"], args.base_file + "_TRAIN.json")
        word_list = build_word_set(input_file, CFG["gpudatadir"], args.max_voc_len)

    print("index loaded.")