import os
import itertools
import pickle as pkl
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
