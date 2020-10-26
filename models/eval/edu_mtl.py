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


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return train(hparams)
    else:
        return main(hparams)


def main(hparams):

    xp_title = hparams.model_type + "_" + hparams.ft_type + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger = init_lightning(xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         logger=logger
                         )

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
                 "hparams": hparams}
    model = EvalModels(**arguments)
    print("Model Loaded.")
    trainer.test(model.cuda(), test_loader)


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


def init_lightning(xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


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
    init(hparams)
