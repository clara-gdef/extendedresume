import glob
import os
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
from utils.model import collate_for_text_gen, collate_for_text_gen_elmo


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
    xp_title = hparams.model_type + "_" + hparams.ft_type + "_" + str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         max_epochs=hparams.epochs,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         logger=logger,
                         auto_lr_find=False
                         )
    datasets = load_datasets(hparams, ["_VALID", "_VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]

    if hparams.ft_type !='elmo':
        collate = collate_for_text_gen
    else:
        collate = collate_for_text_gen_elmo()
    print("Loading word vectors...")
    with open(os.path.join(CFG["gpudatadir"], "tensor_40k_" + hparams.ft_type + ".pkl"), "rb") as f:
        embeddings = pkl.load(f)
    print("Word vectors loaded")

    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate,
                              num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate,
                              num_workers=0)
    print("Dataloaders initiated.")
    arguments = {"embeddings": embeddings,
                 "datadir": CFG["gpudatadir"],
                 "hparams": hparams}

    # print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = FirstJobPredictor(**arguments)
    print("Model Loaded.")
    print("Starting training for model " + xp_title)
    trainer.fit(model.cuda(), train_loader, valid_loader)


def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "input_file": None,
        "index": None,
        "max_seq_length": None,
        "datadir": CFG["gpudatadir"],
        "ft_type": hparams.ft_type,
        "load": (hparams.load_dataset == "True")
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
    parser.add_argument("--hidden_size", type=int, default=100)
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
