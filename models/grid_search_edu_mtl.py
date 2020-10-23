import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train, eval
from utils import DotDict


def grid_search(hparams):
    with ipdb.launch_ipdb_on_exception():
        test_results = {}
        dico = init_args(hparams)
        for str_lr in hparams.lr:
            lr = float(str_lr)
            test_results[lr] = {}
            for b_size in hparams.b_size:
                test_results[lr][int(b_size)] = {}
                print("Grid Search for (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                dico['lr'] = lr
                dico["b_size"] = b_size
                arg = DotDict(dico)
                if hparams.TRAIN == "True":
                    train.edu_mtl.main(arg)
                test_results[lr][b_size] = eval.edu_mtl.main(arg)
            # ## TODO REMOVE THIS - UNINDENT
            res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_" + hparams.model_type + "_topK.pkl")
            with open(res_path, "wb") as f:
                pkl.dump(test_results, f)


def init_args(hparams):
    dico = {'gpus': hparams.gpus,
            "ft_type" : hparams.ft_type,
            'load_dataset': hparams.load_dataset,
            'epochs': hparams.epochs,
            "wd": 0.0,
            "DEBUG": hparams.DEBUG,
            "model_type": hparams.model_type,
            "hidden_size": hparams.hidden_size
            }
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--TRAIN", default="True")
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="e2e_atn")
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--b_size", nargs='+', default=[16, 64, 512])
    parser.add_argument("--lr", nargs='+', default=[1e-1, 1e-2, 1e-3])
    hparams = parser.parse_args()
    grid_search(hparams)
