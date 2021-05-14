import os
import itertools
import pickle as pkl
from collections import Counter
from tqdm import tqdm
import ipdb
import numpy as np
from nltk.tokenize import word_tokenize
# from allennlp.modules.elmo import batch_to_ids
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
            indices.append(index["EOD"])
            break
    actual_len = len(indices)
    while len(indices) < max_seq_length:
        indices.append(0)
    return indices, actual_len


def word_seq_into_list(position, description, index):
    number_regex = re.compile(r'\d+(,\d+)?')
    new_tup = []
    if index is not None:
        new_tup.append("SOT")
        whole_job = position.lower() + " SOD" + ' ' + description.lower()
    else:
        whole_job = position.lower() + ' ' + description.lower()
    job = word_tokenize(whole_job)
    for tok in job:
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        elif tok == "SOD":
            new_tup.append(tok)
        else:
            new_tup.append(tok.lower())
    if index is not None:
        new_tup.append("EOD")
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


def handle_education_ft(edu_list, ft_model):
    sorted_edu_list = sorted(edu_list, key=lambda k: k["to"], reverse=True)
    # keeps 90% of the dataset without trimming experience
    new_ed_tensor = np.zeros((4, ft_model.get_dimension()))
    for num, edu in enumerate(sorted_edu_list):
        if num < 4:
            tokenized_edu = word_seq_into_list(edu["degree"], edu["institution"], None)
            word_count = 0
            tmp = []
            for token in tokenized_edu:
                tmp.append(ft_model.get_word_vector(token))
                word_count += 1
            new_ed_tensor[num, :] = np.mean(np.stack(tmp), axis=0) / word_count
    return new_ed_tensor


def to_elmo_emb(edu_list, elmo):
    sorted_edu_list = sorted(edu_list, key=lambda k: k["to"], reverse=True)
    # keeps 90% of the dataset without trimming experience
    new_ed_tensor = np.zeros((4, 1024))
    tmp = []
    for num, edu in enumerate(sorted_edu_list):
        line = edu["degree"].lower() + ' ' + edu["institution"].lower()
        if num < 4:
            character_ids = batch_to_ids(line)
            emb = elmo(character_ids.cuda())
            tmp.append(np.sum(emb["elmo_representations"][-1].detach().cpu().numpy(), axis=0) / len(line))
            new_ed_tensor[num, :] = np.mean(np.stack(tmp), axis=0)
    return new_ed_tensor


def build_word_count(input_file):
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
                tokenized_edu = word_seq_into_list(edu["degree"], edu["institution"], None)
                for word in tokenized_edu:
                    if re.match(number_regex, word):
                        word_count["NUM"] += 1
                    else:
                        word_count[word] += 1
            pbar.update(1)

    return word_count