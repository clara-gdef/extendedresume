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
import torch
from data.datasets import ProfilesForCamembert
from models.classes.End2EndCamembert import End2EndCamembert
from utils import get_ind_class_dict
from utils.model import collate_for_bert_edu, collate_for_bert_jobs, get_latest_model


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
        persistent_workers = False
    else:
        trainer = pl.Trainer(gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             callbacks=call_back_list,
                             logger=logger,
                             accelerator='ddp_spawn',
                             precision=16
                             )
        num_workers = hparams.num_workers
        persistent_workers = False
    print("Dataloaders initiated.")
    arguments = {'hp': hparams,
                 'desc': xp_title,
                 "model_path": model_path,
                 "datadir": CFG["gpudatadir"]}
    print("Initiating model...")
    model = End2EndCamembert(**arguments)
    print("Model Loaded.")
    if hparams.input_type == "jobs":
        collate = collate_for_bert_jobs
    elif hparams.input_type == "edu":
        collate = collate_for_bert_edu
    else:
        raise Exception(
            "wrong input type, can be either \"job\" or \"edu\", " + str(hparams.input_type) + " was given.")
    if hparams.TRAIN == "True":
        datasets = load_datasets(hparams, ["TRAIN", "TRAIN"])
        dataset_train, dataset_valid = datasets[0], datasets[1]
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate,
                                  num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
        print("Model Loaded.")
        if hparams.load_from_checkpoint == "True":
            print("Loading from previous checkpoint...")
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_file = os.path.join(model_path, f"epoch={hparams.checkpoint}.ckpt")
            model.load_state_dict(torch.load(model_file)["state_dict"])
            print("Resuming training from checkpoint : " + model_file + ".")
        if hparams.auto_lr_find == "True":
            print("looking for best lr...")
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
            # Results can be found in
            print(lr_finder.results)
            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()
            # update hparams of the model
            model.hp.lr = new_lr
            print(f"NEW LR = {new_lr}")
            ipdb.set_trace()
        print("Starting training for " + xp_title + "...")
        trainer.fit(model, train_loader, valid_loader)
    if hparams.TEST == "True":
        model_file = get_latest_model(CFG["modeldir"], model_name)
        if hparams.TRAIN == "False":
            print("Loading from previous run...")
            model.load_state_dict(torch.load(model_file))
        print("Evaluating model : " + model_file + ".")
        datasets = load_datasets(hparams, ["TEST"])
        dataset_test = datasets[0]
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate,
                                 num_workers=0, drop_last=False)
        # model.hp.b_size = 1
        return trainer.test(test_dataloaders=test_loader, model=model)


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
                 "is_toy": hparams.toy_dataset,
                 "load": hparams.load_dataset == "True"}
    for split in splits:
        ipt_file = os.path.join(CFG["gpudatadir"], f"bp_3jobs_desc_edu_skills_industry_date_company_FR_{split}.json")
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
        filename='{epoch}-{val_loss:.2f}',
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
    if hparams.load_from_checkpoint == "True":
        xp_title = f"{hparams.model_type}_{hparams.input_type}_bs{hparams.b_size}_lr{hparams.lr}_{hparams.optim}"
    else:
        xp_title = f"{hparams.model_type}_{hparams.input_type}_{hparams.hidden_size}_bs{hparams.b_size}_lr{hparams.lr}_{hparams.optim}"
    # if hparams.subsample != -1:
    #     xp_title += f"sub{hparams.subsample}"
    if hparams.end2end == "True":
        xp_title += f"_fine_tuned"
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--build_ind_dict", default="False")
    parser.add_argument("--print_preds", type=str, default="True")
    parser.add_argument("--end2end", default="True")
    parser.add_argument("--TEST", type=str, default="False")
    parser.add_argument("--TRAIN", type=str, default="True")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=str, default="29-step=60899")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--toy_dataset", type=str, default="True")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="bert_prof")
    parser.add_argument("--input_type", type=str, default="jobs")  # can be job or edu
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
