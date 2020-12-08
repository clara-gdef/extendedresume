import glob
import os
import ipdb
import argparse
import pickle as pkl

import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import json
import yaml
from data.datasets import AggregatedEduDataset
from models.classes.EvalModels import EvalModels
from utils import get_ind_class_dict
from utils.model import collate_for_edu, get_model_params, get_latest_model


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return main(hparams)
    else:
        return main(hparams)


def main(hparams):
    predictions = get_predictions(hparams.load_dataset)
    ind_classes = get_ind_class_dict(False, CFG)
    rev_ind_class = {v:k for k, v in ind_classes.items()}
    right_ind, wrong_ind = get_right_wrong_pred(predictions, "ind")
    right_pred_file =  os.path.join(CFG["gpudatadir"],"ind_right_pred_edu.txt")
    wrong_pred_file =  os.path.join(CFG["gpudatadir"],"ind_wrong_pred_edu.txt")
    base_file = os.path.join(CFG["gpudatadir"], "bp_3jobs_desc_edu_skills_industry_date_company_FR_TEST.json")
    with open(right_pred_file, 'a+') as rpf:
        with open(wrong_pred_file, 'a+') as wpf:
            with open(base_file, 'r') as f:
                num_lines = sum(1 for line in f)
            with open(base_file, 'r') as f:
                pbar = tqdm(f, total=num_lines, desc="Building tuple for split: TEST")
                for line in f:
                    raw_p = json.loads(line)
                    id_p = raw_p[0]
                    edu = format_profile(raw_p[3])
                    if id_p in right_ind.keys():
                        ipdb.set_trace()
                        rpf.write("ID " + str(id_p) + "===============================================================\n")
                        rpf.write("INDUSTRY " + str(ind_classes[right_ind[id_p]["pred"]]))
                        rpf.write("EDUCATION \n")
                        for num, item in enumerate(edu):
                            rpf.write("(" + num + ")" + item + "\n")
                    elif id_p in wrong_ind.keys():
                        ipdb.set_trace()
                        wpf.write("ID " + str(id_p) + "===============================================================\n")
                        wpf.write("INDUSTRY " + str(ind_classes[wrong_ind[id_p]["lab"]]))
                        wpf.write("PREDICTION " + str(ind_classes[wrong_ind[id_p]["pred"]]))
                        wpf.write("EDUCATION \n")
                        for num, item in enumerate(edu):
                            wpf.write("(" + str(num) + ") : " + item + "\n")
                    else:
                        continue
                    pbar.update(1)


def format_profile(edu_list):
    formated_edu = []
    sorted_edu_list = sorted(edu_list, key=lambda k: k["to"], reverse=True)
    for num, edu in enumerate(sorted_edu_list):
        formated_edu.append(word_seq_into_list(edu["degree"], edu["institution"]))
    return formated_edu

def get_predictions(load):
    tgt_dir = os.path.join(CFG["gpudatadir"], "edu_preds.pkl")
    if load == "True":
        with open(tgt_dir, 'rb') as f:
            predictions = pkl.load(f)
    else:
        xp_title = hparams.model_type + "_" + hparams.ft_type + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
        datasets = load_datasets(hparams, ["TEST"])
        dataset_test = datasets[0]

        in_size, hidden_size, num_class_sk, num_class_ind = get_model_params(hparams, dataset_test)
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_edu,
                                  num_workers=8, shuffle=True)

        print("Dataloaders initiated.")
        arguments = {'input_size': in_size,
                     'hidden_size': hidden_size,
                     "num_classes_skills": num_class_sk,
                     "num_classes_ind": num_class_ind,
                     "datadir": CFG["gpudatadir"],
                     "hparams": hparams}
        model = EvalModels(**arguments)
        latest_file = get_latest_model(CFG["modeldir"], xp_title)
        print("Evaluating model " + latest_file)
        model.load_state_dict(torch.load(latest_file)["state_dict"])
        model = model.cuda()
        print("Model Loaded.")
        predictions = model.get_outputs(test_loader)
        with open(tgt_dir, 'wb') as f:
            pkl.dump(predictions, f)

    return predictions



def get_right_wrong_pred(predictions, handle):
    right_preds = {}
    wrong_preds = {}
    for i in predictions.keys():
        if predictions[i][handle + "_lab"] == predictions[i][handle + "_pred"]:
            right_preds[i] = predictions[i][handle + "_lab"]
        else:
            wrong_preds[i] = {"lab": predictions[i][handle + "_lab"] , "pred": predictions[i][handle + "_pred"] }
    return right_preds, wrong_preds



def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "ft_type": hparams.ft_type,
        "load": (hparams.load_dataset == "True")
    }
    for split in splits:
        if hparams.ft_type !=  "elmo":
            common_hparams["input_file"] = os.path.join(CFG["gpudatadir"], "flat_profiles_dataset_" + split + ".pkl")
        else:
            common_hparams["input_file"] = os.path.join(CFG["gpudatadir"], "flat_profiles_dataset_elmo_" + split + ".pkl")
        datasets.append(AggregatedEduDataset(**common_hparams, split=split))

    return datasets


def word_seq_into_list(position, description):
    number_regex = re.compile(r'\d+(,\d+)?')
    new_tup = []
    whole_job = "DEGREE: " + position.lower() + ', INSTITUTION: ' + description.lower()
    for tok in whole_job.split(" "):
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        elif tok == "DEGREE" or tok == "INSTITUTION":
            new_tup.append(tok)
        else:
            new_tup.append(tok.lower())
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='elmo')
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="edu_mtl")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
