import glob
import os
from allennlp.modules.elmo import Elmo
import pickle as pkl
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import TextGenerationDataset
from models.classes.FirstJobPredictor import FirstJobPredictor
from utils.model import collate_for_text_gen, collate_for_text_gen_elmo, get_latest_model


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
    xp_title = hparams.model_type + "_" + hparams.ft_type + "_" + str(hparams.b_size) + "_" + str(hparams.lr) + "_" + str(hparams.hidden_size) + '_' + str(hparams.wd)
    logger = init_lightning(hparams, xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         max_epochs=hparams.epochs,
                         logger=logger
                         )
    datasets = load_datasets(hparams, ["TEST"])
    dataset_test = datasets[0]

    if hparams.ft_type != 'elmo':
        collate = collate_for_text_gen
    else:
        collate = collate_for_text_gen_elmo

    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate, num_workers=0)
    print("Dataloaders initiated.")
    arguments = {"dim": get_emb_dim(hparams),
                 "index": dataset_test.index,
                 "datadir": CFG["gpudatadir"],
                 "hparams": hparams,
                 "class_weights": None,
                 "elmo": None}
    if hparams.ft_type == "elmo":
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        elmo = Elmo(options_file, weight_file, 2, dropout=0)
        arguments["elmo"] = elmo.cuda()

    model = FirstJobPredictor(**arguments)

    latest_file = get_latest_model(CFG["modeldir"], xp_title)
    print("Evaluating model " + latest_file)
    return trainer.test(model.cuda(), test_loader)


def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "input_file": None,
        "index": None,
        "embedder": None,
        "max_seq_length": None,
        "datadir": CFG["gpudatadir"],
        "ft_type": hparams.ft_type,
        "load": (hparams.load_dataset == "True"),
        "subsample": hparams.subsample
    }
    for split in splits:
        datasets.append(TextGenerationDataset(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, xp_title):
    model_path = os.path.join(CFG['modeldir'], xp_title)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


def get_emb_dim(hparams):
    if hparams.ft_type == "elmo":
        dim = 1024
    else:
        dim = 300
    return dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="fj_gen")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--dpo", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
