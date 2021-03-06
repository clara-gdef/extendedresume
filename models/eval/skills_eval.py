import argparse
import os
import pickle as pkl
import ipdb
import torch
import yaml
from tqdm import tqdm
import numpy as np
from utils.model import get_preds_wrt_threshold, get_metrics_for_skills


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        tgt_file = os.path.join(CFG["gpudatadir"],
                                "outputs_eval_models_" + args.model_type + "_" + args.ft_type + "_" + str(
                                    args.b_size) + "_" + str(args.lr) + ".pkl")
        with open(tgt_file, "rb") as f:
            res_dict = pkl.load(f)
        print("Data loaded.")
        sk_preds = torch.stack(res_dict["sk"]["preds"])
        sk_labels = torch.stack(res_dict["sk"]["labels"])
        res = {}
        # lower_bound = torch.min(sk_preds).item() / 3
        # higher_bound = torch.max(sk_preds).item() + (torch.max(sk_preds).item() / 3)
        # print("Lower bound : " + str(lower_bound))
        # print("Higher bound : " + str(higher_bound))
        # for threshold in tqdm(np.linspace(lower_bound, higher_bound, 10), desc="evaluating skills..."):
        #     new_preds = get_preds_wrt_threshold(sk_preds, threshold)
        #     res[threshold] = get_metrics_for_skills(new_preds.squeeze(1).cpu().numpy(), sk_labels.squeeze(1).cpu().numpy(), 523, "skills")
        th = 0.000227824674059067
        print("Computing pred wrt threshold...")
        new_preds = get_preds_wrt_threshold(sk_preds, th)
        print("Computing metrics wrt threshold...")
        res[th] = get_metrics_for_skills(new_preds.squeeze(1).cpu().numpy(), sk_labels.squeeze(1).cpu().numpy(), 523, "skills")
        print(res)
        ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="edu_mtl")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0.0)
    hparams = parser.parse_args()
    main(hparams)
