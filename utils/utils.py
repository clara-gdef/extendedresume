import re
import ipdb
import torch
from nltk.tokenize import word_tokenize


def word_seq_into_list(position, description):
    number_regex = re.compile(r'\d+(,\d+)?')
    new_tup = []
    whole_job = position.lower() + ' ' + description.lower()
    job = word_tokenize(whole_job)
    for tok in job:
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        else:
            new_tup.append(tok.lower())
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


def collate_for_flat_profiles(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    edu = [i[2] for i in batch]
    skills = [i[3] for i in batch]
    ind = [i[4] for i in batch]
    return ids, torch.stack(torch.from_numpy(jobs)), torch.stack(torch.from_numpy(edu)), skills, ind


def get_model_params(args, dataset):
    return 300, args.hidden_size, len(dataset.skills_classes), len(dataset.ind_classes)