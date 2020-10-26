import glob
import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import FlatProfilesDataset
from models.classes.End2EndProfileBuilder import End2EndProfileBuilder
from utils.model import collate_for_flat_profiles, get_model_params


def init(hparams):
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return main(hparams)
    else:
        return main(hparams)


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    xp_title = hparams.model_type + "_" + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger = init_lightning(xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         logger=logger,
                         auto_lr_find=False
                         )
    datasets = load_datasets(["TEST"])
    dataset_test = datasets[0]

    in_size, hidden_size, num_class_sk, num_class_ind = get_model_params(hparams, dataset_test)
    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_flat_profiles,
                              num_workers=8)
    print("Dataloaders initiated.")
    arguments = {'input_size': in_size,
                 'hidden_size': hidden_size,
                 "num_classes_skills": num_class_sk,
                 "num_classes_ind": num_class_ind,
                 "hparams": hparams}

    # print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = End2EndProfileBuilder(**arguments)

    model_path = os.path.join(CFG['modeldir'], xp_title)
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    print("Evaluating model " + latest_file)
    model.load_state_dict(torch.load(latest_file)["state_dict"])
    print("Model Loaded.")
    return trainer.test(model.cuda(), test_loader)


def load_datasets(splits):
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "input_file": None,
        "ft_job": None,
        "ft_edu": None,
        "skills_classes": None,
        "ind_classes": None,
        "load": True
    }
    for split in splits:
        datasets.append(FlatProfilesDataset(**common_hparams, split=split))

    return datasets


def init_lightning(xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="e2e_atn")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    hparams = parser.parse_args()
    init(hparams)
