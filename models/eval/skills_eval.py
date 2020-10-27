import argparse
import os
import pickle as pkl
import ipdb
import yaml
import numpy as np
from utils.model import get_preds_wrt_threshold, get_metrics


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    tgt_file = os.path.join(CFG["gpudatadir"],
                            "outputs_eval_models_" + args.model_type + "_" + args.lr + "_" + args.b_size + ".pkl")
    with open(tgt_file, "wb") as f:
        res_dict = pkl.load(f)

    sk_preds = res_dict["sk"]["preds"]
    sk_labels = res_dict["sk"]["labels"]
    ipdb.set_trace()
    res = {}
    for threshold in np.linspace(0, 1, 10):
        new_preds = get_preds_wrt_threshold(sk_preds, round(threshold, 1))
        res[round(threshold, 1)] = get_metrics(new_preds.squeeze(1).cpu().numpy(), sk_labels.squeeze(1).cpu().numpy(), 523, "skills")
    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="edu")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    hparams = parser.parse_args()
    main(hparams)
