import glob
import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import pickle as pkl
from data.datasets import ProfilesForCamembert
from models.classes.EvalModels import EvalModels
from utils import get_ind_class_dict
from utils.model import collate_for_edu, get_model_params


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
    xp_title = make_xp_title(hparams)
    model_name = "/".join(xp_title.split('_'))
    logger, checkpoint_callback, early_stop_callback, model_path = init_lightning(xp_title, model_name)
    call_back_list = [checkpoint_callback, early_stop_callback]
    if hparams.DEBUG == "True":
        trainer = pl.Trainer(gpus=1,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             precision=16
                             )
        num_workers = 0
    else:
        trainer = pl.Trainer(gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             # gradient_clip_val=hparams.clip_val,
                             accelerator='ddp_spawn',
                             precision=16
                             )
    datasets = load_datasets(hparams, ["TRAIN", "VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]


def load_datasets(hparams, splits):
    datasets = []
    with open(os.path.join(CFG["gpudatadir"], "good_skills.p"), 'rb') as f:
        skills = pkl.load(f)
    skills_classes = {k: v for k, v in enumerate(sorted(skills))}
    ind_classes = get_ind_class_dict(hparams.build_ind_dict, CFG)
    arguments = {'datadir': CFG["gpudatadir"],
                 "subsample": hparams.subsample,
                 "skills_classes": skills_classes,
                 "ind_classes": ind_classes,
                 "load": hparams.load_dataset == "True"}
    for split in splits:
        ipt_file = os.path.join(CFG["gpudatadir"], f"bp_3jobs_desc_edu_skills_industry_date_company_FR_{split}_.json")
        tmp = ProfilesForCamembert(**arguments,
                                   input_file=ipt_file,
                                   split=split)
        datasets.append(tmp)
    return datasets


def init_lightning(xp_title, model_name):
    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    print("callback initiated.")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback, model_path


def make_xp_title(hparams):
    xp_title = f"{hparams.model_type}_bs{hparams.b_size}_lr{hparams.lr}_{hparams.optim}"
    if hparams.subsample != -1:
        xp_title += f"sub{hparams.subsample}"
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--load_dataset", default="False")
    parser.add_argument("--optim", type=str, default="adam")
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
