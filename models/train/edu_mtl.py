import glob
import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import AggregatedEduDataset
from models.classes.EvalModels import EvalModels
from utils.utils import collate_for_edu, get_model_params


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return train(hparams)
    else:
        return train(hparams)


def train(hparams):
    xp_title = hparams.model_type + "_" + hparams.ft_type + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         max_epochs=hparams.epochs,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         logger=logger,
                         auto_lr_find=False
                         )
    #TODO : replace by TRAIN
    datasets = load_datasets(hparams, ["TRAIN", "VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]

    in_size, hidden_size, num_class_sk, num_class_ind = get_model_params(hparams, dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_edu,
                              num_workers=8, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_edu,
                              num_workers=8)
    print("Dataloaders initiated.")
    arguments = {'input_size': in_size,
                 'hidden_size': hidden_size,
                 "num_classes_skills": num_class_sk,
                 "num_classes_ind": num_class_ind,
                 "hparams": hparams}

    # print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = EvalModels(**arguments)
    print("Model Loaded.")
    trainer.fit(model.cuda(), train_loader, valid_loader)


def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "ft_type": hparams.ft_type,
        "load": (hparams.load_dataset == "True")
    }
    for split in splits:
        common_hparams["input_file"] = os.path.join(CFG["gpudatadir"], "flat_profiles_dataset_" + split + ".pkl")
        datasets.append(AggregatedEduDataset(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, xp_title):
    model_name = hparams.model_type + "_" + hparams.ft_type + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_path, '{epoch:02d}'),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    print("callback initiated.")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.000,
        patience=10,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="edu")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)
