import json
import os
import pickle as pkl
import argparse
import torch
import yaml
from tqdm import tqdm
from allennlp.modules.elmo import Elmo
import fastText
from utils import word_seq_into_list
import ipdb
from allennlp.modules.elmo import batch_to_ids
from collections import Counter
import re


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        input_file = os.path.join(CFG["gpudatadir"], args.base_file + "_TRAIN.json")

        if args.build_vocab == "True":
            word_list = build_word_set(input_file, args)
        else:
            with open(os.path.join(CFG["gpudatadir"], "vocab_40k.pkl"), "rb") as f:
                word_list = pkl.load(f)

        if args.ft_type == 'fs':
            embedder = fastText.load_model(os.path.join(CFG["prevmodeldir"], "ft_fs.bin"))
        elif args.ft_type == 'pt':
            embedder = fastText.load_model(os.path.join(CFG["modeldir"],  "ft_pre_trained.bin"))
        elif args.ft_type == "elmo":
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            elmo = Elmo(options_file, weight_file, 2, dropout=0)
            embedder = elmo.cuda()

        build_index_and_tensor(word_list, embedder, args)


def build_word_set(input_file, args):
    word_count = Counter()
    number_regex = re.compile(r'\d+(,\d+)?')
    with open(input_file, 'r') as f:
        num_lines = sum(1 for line in f)
    with open(input_file, 'r') as f:
        pbar = tqdm(f, total=num_lines, desc="Building vocab from Train split...")
        for line in f:
            data = json.loads(line)
            sorted_edu = sorted(data[-2], key=lambda k: k['to'], reverse=True)
            for edu in sorted_edu:
                tokenized_edu = word_seq_into_list(edu["degree"], edu["institution"])
                for word in tokenized_edu:
                    if re.match(number_regex, word):
                        word_count["NUM"] += 1
                    else:
                        word_count[word] += 1
            pbar.update(1)

    word_list = [x[0] for x in word_count.most_common(args.max_voc_len)]

    with open(os.path.join(CFG["gpudatadir"], "vocab_40k.pkl"), "wb") as f:
        pkl.dump(word_list, f)
    return word_list


def build_index_and_tensor(word_list, embedder, args):
    word_to_index = dict()
    print("Length of the vocabulary: " + str(len(word_list)))
    if args.ft_type != "elmo":
        dim = 300
    else:
        dim = 1024
    tensor_updated, w2i_updated, num_tokens = build_special_tokens(word_to_index, dim)
    with tqdm(total=len(word_list), desc="Building tensors and index...") as pbar:
        for i, word in enumerate(word_list):
            if args.ft_type != "elmo":
                tensor_updated = torch.cat([tensor_updated, torch.FloatTensor(embedder.get_word_vector(word)).view(1, -1)], dim=0)
            else:
                character_ids = batch_to_ids(word)
                emb = embedder(character_ids.cuda())
                tensor_updated = torch.cat([tensor_updated, torch.mean(emb["elmo_representations"][-1], dim=0).view(1, -1).detach().cpu()], dim=0)
            w2i_updated[word] = i + num_tokens
            pbar.update(1)
    print(len(word_to_index))
    with open(os.path.join(CFG["gpudatadir"], "tensor_40k_" + args.ft_type + ".pkl"), "wb") as f:
        pkl.dump(tensor_updated, f)
    with open(os.path.join(CFG["gpudatadir"], "index_40k_" + args.ft_type + ".pkl"), "wb") as f:
        pkl.dump(w2i_updated, f)


def build_special_tokens(word_to_index, dim):
    """
    SOT stands for 'start of title'
    EOT stands for 'end of title'
    SOD stands for 'start of degree'
    SOI stands for 'start of institution'
    PAD stands for 'padding index'
    UNK stands for 'unknown word'
    """
    SOD = torch.randn(1, dim)
    SOI = torch.randn(1, dim)
    EOI = torch.randn(1, dim)
    PAD = torch.randn(1, dim)
    UNK = torch.randn(1, dim)
    word_to_index["PAD"] = 0
    tensor = PAD
    word_to_index["SOD"] = 1
    tensor = torch.cat([tensor, SOD], dim=0)
    word_to_index["SOI"] = 2
    tensor = torch.cat([tensor, SOI], dim=0)
    word_to_index["EOI"] = 3
    tensor = torch.cat([tensor, EOI], dim=0)
    word_to_index["UNK"] = 4
    tensor = torch.cat([tensor, UNK], dim=0)
    return tensor, word_to_index, 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--build_vocab", type=str, default="False")
    parser.add_argument("--max_voc_len", type=int, default=40000)
    parser.add_argument("--min_occurence", type=int, default=5)
    args = parser.parse_args()
    main(args)

