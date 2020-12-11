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
        index_file = os.path.join(CFG["gpudatadir"], "index_40k.pkl")
        with open(index_file, 'rb') as f:
            index = pkl.load(f)
        print("index loaded.")

        load = (args.load_dataset == "True")

        print("Loading word vectors...")
        if args.ft_type == 'elmo':
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            elmo = Elmo(options_file, weight_file, 2, dropout=0)
            embedder = elmo.cuda()
        elif args.ft_type == 'fs':
            embedder = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs_edu_job.bin"))
        elif args.ft_type == 'pt':
            embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_pre_trained.bin"))
        print("Word vectors loaded.")

        for split in ["TEST", "VALID", "TRAIN"]:
            input_file = os.path.join(CFG["gpudatadir"], args.base_file + "_" + split + ".json")
            TextGenerationDataset(CFG["gpudatadir"], input_file, index, split, args.ft_type, args.max_seq_length, embedder, 0, load)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--load_dataset", type=str, default="False")
    args = parser.parse_args()
    main(args)
