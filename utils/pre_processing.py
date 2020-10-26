import os
import itertools
import pickle as pkl

import ipdb
from nltk.tokenize import word_tokenize
import re
import json


def build_ind_class_dict(CFG):
    print("Building industry class dict...")
    input_files = []
    for split in ["_TEST", "_VALID", "_TRAIN"]:
        input_files.append(os.path.join(CFG["prevdatadir"], args.base_file + split + ".json"))

    classes = set()
    for filename in itertools.chain(input_files):
        with open(filename, "r") as f:
            for line in f:
                person = json.loads(line)
                classes.add(person[-1])

    class_dict = {}
    for num, industry in enumerate(sorted(classes)):
        class_dict[num] = industry
    print("Done.")
    return class_dict


def get_ind_class_dict(build_ind_dict, CFG):
    if build_ind_dict == "True":
        class_dict = build_ind_class_dict()
        with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'wb') as f:
            pkl.dump(class_dict, f)
    else:
        with open(os.path.join(CFG["gpudatadir"], "ind_class_dict.pkl"), 'rb') as f:
            class_dict = pkl.load(f)
    return class_dict


def word_list_to_indices(word_list, index, max_seq_length):
    indices = []
    for word in word_list:
        if len(indices) < max_seq_length - 1:
            if word in index.keys():
                indices.append(index[word])
            else:
                indices.append(index["UNK"])
        else:
            indices.append(index["EOI"])
            break
    return indices


def word_seq_into_list(position, description, index):
    number_regex = re.compile(r'\d+(,\d+)?')
    new_tup = []
    if index is not None:
        new_tup.append("SOD")
        whole_job = position.lower() + "SOI" + ' ' + description.lower()
    else:
        whole_job = position.lower() + ' ' + description.lower()
    job = word_tokenize(whole_job)
    for tok in job:
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        else:
            new_tup.append(tok.lower())
    if index is not None:
        new_tup.append("EOI")
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup

