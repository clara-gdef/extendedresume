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
    xp_title = hparams.model_type + "_" + hparams.ft_type + "_" + str(hparams.b_size) + "_" + str(hparams.lr) + "_" + str(hparams.hidden_size) + '_' + str(hparams.wd)
    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         max_epochs=hparams.epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         logger=logger
                         )
    ### TODO remove double train to replace with valid
    datasets = load_datasets(hparams, ["TRAIN", "VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]

    with open(os.path.join(CFG["gpudatadir"], "token_frequencies.pkl"), "rb") as f:
        frqc = pkl.load(f)

    tmp = dataset_train.tuples
    for i in range(1):
        dataset_train.tuples.extend(tmp)

    if hparams.ft_type != 'elmo':
        collate = collate_for_text_gen
    else:
        collate = collate_for_text_gen_elmo

    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate,
                              num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate,
                              num_workers=0)
    print("Dataloaders initiated.")
    arguments = {"dim": get_emb_dim(hparams),
                 "index": dataset_train.index,
                 "class_weights": frqc,
                 "datadir": CFG["gpudatadir"],
                 "hparams": hparams,
                 "elmo": None}
    if hparams.ft_type == "elmo":
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        elmo = Elmo(options_file, weight_file, 2, dropout=0)
        arguments["elmo"] = elmo.cuda()

    model = FirstJobPredictor(**arguments)

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # #fig = lr_finder.plot(suggest=True)
    # #fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    #
    # # update hparams of the model
    # model.hparams.lr = new_lr
    # ipdb.set_trace()

    print("Model Loaded.")
    print("Starting training for model " + xp_title)
    trainer.fit(model.cuda(), train_loader, valid_loader)


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
        min_delta=0.00000,
        patience=hparams.epochs / 5,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback


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
    parser.add_argument("--b_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="fj_gen")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--dpo", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
